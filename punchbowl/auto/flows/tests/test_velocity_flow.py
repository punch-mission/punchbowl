import json
import math
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from punchbowl.auto.control.db import File
from punchbowl.auto.flows.velocity import (
    DEFAULT_PRODUCT_CODE,
    TRACK_VELOCITY_DEFAULTS,
    level3_VAM_construct_file_info,
    level3_VAM_construct_flow_info,
    level3_VAM_query_ready_files,
)
from punchbowl.level3.velocity import get_cadence_min, get_required_frame_count

T0 = datetime(2026, 1, 1, 0, 0, 0)


@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    # Only the files table is needed; the flows table uses MySQL-only column types
    File.__table__.create(engine)
    with Session(engine) as session:
        yield session


def _pipeline_config(**l3_vam_config):
    return {"file_version": "1", "flows": {"level3_VAM": {"t0": T0.isoformat(), **l3_vam_config}}}


def _frame_times(product_code, start, end):
    """Observation times at the product cadence from start to end."""
    cadence = timedelta(minutes=get_cadence_min(product_code))
    return [start + i * cadence for i in range(math.ceil((end - start) / cadence))]


def _add_files(session, product_code, start, end, skip=()):
    """Add files at the product cadence from start to end, except the frame indices in skip."""
    for i, t in enumerate(_frame_times(product_code, start, end)):
        if i not in skip:
            session.add(File(level="3", file_type=product_code[:2], observatory=product_code[2], state="created",
                             file_version="1", software_version="none", polarization="C", date_obs=t))
    session.commit()


def _n_required(product_code):
    n_required, _, _ = get_required_frame_count(product_code, TRACK_VELOCITY_DEFAULTS["flow_window_hours"],
                                                TRACK_VELOCITY_DEFAULTS["time_avg_hours"])
    return n_required


# Frame times are offset from the nominal reference times, as for actual observations
OFFSET = timedelta(minutes=1, seconds=13)
DATA_START = T0 + OFFSET
DATA_END = T0 + timedelta(hours=36)

# Supported products with the longest run of consecutive missing frames tolerated in the flow window,
# from punchbowl.level3.velocity._MISSING_FRAME_RULES
PRODUCTS_MAX_MISSING_RUN = [("PTM", 2), ("CAM", 1)]
PRODUCT_CODES = [product_code for product_code, _ in PRODUCTS_MAX_MISSING_RUN]


def _nearest_frame_index(product_code, hours):
    """Index of the frame from DATA_START nearest to T0 + hours."""
    nominal_time = T0 + timedelta(hours=hours)
    times = _frame_times(product_code, DATA_START, DATA_END)
    return min(range(len(times)), key=lambda i: abs(times[i] - nominal_time))


def _nearest_frame_time(product_code, hours):
    """Time of the frame from DATA_START nearest to T0 + hours, i.e. the reference time of the map at that time."""
    return _frame_times(product_code, DATA_START, DATA_END)[_nearest_frame_index(product_code, hours)]


@pytest.mark.parametrize("product_code", PRODUCT_CODES)
def test_groups_at_frames_nearest_to_nominal_reference_times(session, product_code):
    _add_files(session, product_code, DATA_START, DATA_END)
    groups = level3_VAM_query_ready_files.fn(session, _pipeline_config(product_code=product_code))

    # 12-hour inputs centered on 09:00, 15:00, 21:00 and 03:00 the next day fit in the 36 hours of data
    nominal_hours = (27, 21, 15, 9)
    assert len(groups) == len(nominal_hours)
    for hours, group in zip(nominal_hours, groups):
        # The reference time is the observation time of the frame nearest to the nominal time
        assert group.reference_time == _nearest_frame_time(product_code, hours)
        assert group.reference_time in [f.date_obs for f in group]
        assert len(group) == _n_required(product_code)
    # Consecutive input windows overlap, so some files feed more than one map
    assert set(groups[0]) & set(groups[1])


@pytest.mark.parametrize("product_code", PRODUCT_CODES)
def test_existing_velocity_maps_are_skipped(session, product_code):
    _add_files(session, product_code, DATA_START, DATA_END)
    pipeline_config = _pipeline_config(product_code=product_code)
    groups = level3_VAM_query_ready_files.fn(session, pipeline_config)
    skipped = groups[0].reference_time
    session.add(File(level="3", file_type="VA", observatory="M", state="created", file_version="1",
                     software_version="none", polarization="Y", date_obs=skipped))
    session.commit()

    new_groups = level3_VAM_query_ready_files.fn(session, pipeline_config)
    assert skipped not in [g.reference_time for g in new_groups]
    assert len(new_groups) == len(groups) - 1


@pytest.mark.parametrize(("product_code", "max_run"), PRODUCTS_MAX_MISSING_RUN)
def test_tolerated_missing_frames_are_scheduled(session, product_code, max_run):
    # The longest tolerated run of consecutive missing frames just before 15:00, in the flow window of that map
    i_15 = _nearest_frame_index(product_code, 15)
    _add_files(session, product_code, DATA_START, DATA_END, skip=range(i_15 - max_run, i_15))
    groups = level3_VAM_query_ready_files.fn(session, _pipeline_config(product_code=product_code))
    group = next(g for g in groups if g.reference_time == _nearest_frame_time(product_code, 15))
    assert len(group) == _n_required(product_code) - max_run


@pytest.mark.parametrize(("product_code", "max_run"), PRODUCTS_MAX_MISSING_RUN)
def test_too_many_missing_frames_are_not_scheduled(session, product_code, max_run):
    # One more consecutive missing frame than tolerated just before 15:00, in the flow window of that map
    i_15 = _nearest_frame_index(product_code, 15)
    _add_files(session, product_code, DATA_START, DATA_END, skip=range(i_15 - max_run - 1, i_15))
    groups = level3_VAM_query_ready_files.fn(session, _pipeline_config(product_code=product_code))
    reference_times = [g.reference_time for g in groups]
    assert _nearest_frame_time(product_code, 15) not in reference_times
    # The maps whose flow windows do not hold the missing frames are still scheduled
    assert _nearest_frame_time(product_code, 27) in reference_times


@pytest.mark.parametrize("product_code", PRODUCT_CODES)
def test_product_code_from_config_selects_input_files(session, product_code):
    for code in PRODUCT_CODES:
        _add_files(session, code, T0, DATA_END)
    groups = level3_VAM_query_ready_files.fn(session, _pipeline_config(product_code=product_code))
    assert groups
    assert all(f.file_type == product_code[:2] and f.observatory == product_code[2] for g in groups for f in g)


def test_default_product_code(session):
    for code in PRODUCT_CODES:
        _add_files(session, code, T0, DATA_END)
    pipeline_config = _pipeline_config()
    group = level3_VAM_query_ready_files.fn(session, pipeline_config)[0]
    assert all(f.file_type == DEFAULT_PRODUCT_CODE[:2] and f.observatory == DEFAULT_PRODUCT_CODE[2] for f in group)
    output_file = level3_VAM_construct_file_info.fn(group, pipeline_config)[0]
    call_data = json.loads(level3_VAM_construct_flow_info.fn(group, output_file, pipeline_config).call_data)
    assert call_data["product_code"] == DEFAULT_PRODUCT_CODE


def test_unsupported_product_code_raises(session):
    _add_files(session, "PTM", T0, DATA_END)
    with pytest.raises(ValueError):
        level3_VAM_query_ready_files.fn(session, _pipeline_config(product_code="PIM"))


@pytest.mark.parametrize("product_code", PRODUCT_CODES)
def test_flow_and_file_info_use_group_reference_time(session, product_code):
    _add_files(session, product_code, T0, DATA_END)
    pipeline_config = _pipeline_config(product_code=product_code)
    group = level3_VAM_query_ready_files.fn(session, pipeline_config)[0]

    output_file = level3_VAM_construct_file_info.fn(group, pipeline_config, reference_time=None)[0]
    assert output_file.date_obs == group.reference_time

    flow = level3_VAM_construct_flow_info.fn(group, output_file, pipeline_config, reference_time=None)
    call_data = json.loads(flow.call_data)
    assert datetime.fromisoformat(call_data["reference_time"]) == group.reference_time
    assert call_data["product_code"] == product_code
    assert len(call_data["files"]) == len(group)
    # With no flow tracking parameter set in the pipeline configuration, all take the track_velocity defaults
    for key, default in TRACK_VELOCITY_DEFAULTS.items():
        expected = list(default) if isinstance(default, tuple) else default
        assert call_data[key] == expected


@pytest.mark.parametrize("product_code", PRODUCT_CODES)
def test_flow_parameters_from_config_override_defaults(session, product_code):
    _add_files(session, product_code, T0, DATA_END)
    pipeline_config = _pipeline_config(product_code=product_code, delta_t_minutes=96, annuli_centers_rs=[20, 40, 60])
    group = level3_VAM_query_ready_files.fn(session, pipeline_config)[0]
    output_file = level3_VAM_construct_file_info.fn(group, pipeline_config)[0]
    call_data = json.loads(level3_VAM_construct_flow_info.fn(group, output_file, pipeline_config).call_data)
    assert call_data["delta_t_minutes"] == 96
    assert call_data["annuli_centers_rs"] == [20, 40, 60]
