import os
import itertools
from datetime import UTC, datetime, timedelta

from freezegun import freeze_time
from prefect.logging import disable_run_logger
from prefect.testing.utilities import prefect_test_harness
from pytest_mock_resources import create_mysql_fixture

from punchbowl import __version__
from punchbowl.auto.control.db import Base, File, Flow
from punchbowl.auto.control.util import batched, load_pipeline_configuration
from punchbowl.auto.flows.level2 import (
    group_l2_inputs,
    group_l2_inputs_single_observatory,
    level2_construct_file_info,
    level2_construct_flow_info,
    level2_query_ready_clear_files,
    level2_query_ready_files,
    level2_scheduler_flow,
)

TEST_DIR = os.path.dirname(__file__)


def session_fn(session):
    level0_fileM = File(level='0',
                        file_type='PM',
                        observatory='3',
                        state='progressed',
                        file_version='none',
                        software_version='none',
                        polarization='M',
                        date_obs=datetime(2023, 1, 1, 0, 0, 0),
                        date_created=datetime(2022, 12, 25, 0, 0, 0))

    level0_fileZ = File(level='0',
                        file_type='PZ',
                        observatory='3',
                        state='progressed',
                        file_version='none',
                        software_version='none',
                         polarization='Z',
                        date_obs=datetime(2023, 1, 1, 0, 0, 0),
                        date_created=datetime(2022, 12, 25, 0, 0, 0))

    level0_fileP = File(level='0',
                        file_type='PP',
                        observatory='3',
                        state='progressed',
                        file_version='none',
                        software_version='none',
                        polarization='P',
                        date_obs=datetime(2023, 1, 1, 0, 0, 0),
                        date_created=datetime(2022, 12, 25, 0, 0, 0))

    level1_fileM = File(level='1',
                       file_type='PM',
                       observatory='3',
                       state='created',
                       file_version='none',
                       software_version='none',
                       polarization='M',
                       date_obs=datetime(2023, 1, 1, 0, 2, 0),
                       date_created=datetime(2022, 12, 25, 0, 0, 0))

    level1_fileZ = File(level='1',
                       file_type='PZ',
                       observatory='3',
                       state='created',
                       file_version='none',
                       software_version='none',
                       polarization='Z',
                       date_obs=datetime(2023, 1, 1, 0, 1, 0),
                       date_created=datetime(2022, 12, 25, 0, 0, 0))

    level1_fileP = File(level='1',
                       file_type='PP',
                       observatory='3',
                       state='created',
                       file_version='none',
                       software_version='none',
                       polarization='P',
                       date_obs=datetime(2023, 1, 1, 0, 0, 0),
                       date_created=datetime(2022, 12, 25, 0, 0, 0))

    level0_file_clear = File(level='0',
                             file_type='CR',
                             observatory='3',
                             state='progressed',
                             file_version='none',
                             software_version='none',
                             polarization='C',
                             date_obs=datetime(2023, 1, 1, 0, 0, 0),
                             date_created=datetime(2022, 12, 25, 0, 0, 0))

    level1_file_clear = File(level='1',
                             file_type='CR',
                             observatory='3',
                             state='created',
                             file_version='none',
                             software_version='none',
                             polarization='C',
                             date_obs=datetime(2023, 1, 1, 0, 0, 0),
                             date_created=datetime(2022, 12, 25, 0, 0, 0))

    level1_file_clear_not_ready = File(level='1',
                             file_type='CR',
                             observatory='3',
                             state='creating',
                             file_version='none',
                             software_version='none',
                             polarization='C',
                             date_obs=datetime(2023, 1, 1, 0, 0, 1),
                             date_created=datetime(2022, 12, 25, 0, 0, 0))

    session.add(level0_fileM)
    session.add(level0_fileZ)
    session.add(level0_fileP)
    session.add(level1_fileM)
    session.add(level1_fileZ)
    session.add(level1_fileP)
    session.add(level0_file_clear)
    session.add(level1_file_clear)
    session.add(level1_file_clear_not_ready)


db = create_mysql_fixture(Base, session_fn, session=True)


def test_level2_query_ready_files():
    """
    Ensures that polarized inputs across all observatories are grouped correctly for every possible combination of "is
    this observatory's PZM triplet complete?"
    """
    # First we generate input files---three PZM triplets for each observatory
    wfi1, wfi2, wfi3, nfi = [], [], [], []
    t0 = datetime(2025, 6, 1, 1)
    for group_dt in [0, 4, 8]:
        for p, dt in zip(['P', 'Z', 'M'], [0, 65, 130]):
            wfi1.append(File(level='1', file_type=f"P{p}", observatory='1', file_version='1', software_version='1',
                             date_obs=t0 + timedelta(minutes=group_dt, seconds=dt), state='created', polarization=p))
        for p, dt in zip(['P', 'Z', 'M'], [5, 62, 136]):
            wfi2.append(File(level='1', file_type=f"P{p}", observatory='2', file_version='1', software_version='1',
                             date_obs=t0 + timedelta(minutes=group_dt, seconds=dt), state='created', polarization=p))
        for p, dt in zip(['P', 'Z', 'M'], [3, 66, 128]):
            wfi3.append(File(level='1', file_type=f"P{p}", observatory='3', file_version='1', software_version='1',
                             date_obs=t0 + timedelta(minutes=group_dt, seconds=dt), state='created', polarization=p))
        for p, dt in zip(['M', 'Z', 'P'], [7, 72, 139]):
            nfi.append(File(level='1', file_type=f"P{p}", observatory='4', file_version='1', software_version='1',
                            date_obs=t0 + timedelta(minutes=group_dt, seconds=dt), state='created', polarization=p))

    with disable_run_logger():
        with freeze_time(datetime(2025, 6, 2, 0, 0, 0)) as frozen_datatime:  # noqa: F841
            pipeline_config = {'flows': {'level2': {'ignore_missing_after_days': 0.5}}}

            file_to_exclude = 0
            # There are 12 MZP groups in total. For each we make a [True, False] pair (for "is this triplet complete?"),
            # and we iterate through every combination of choices from each of those 12 sets.
            for groups_are_complete in list(itertools.product(*([[True, False]] * 12))):
                input_files = []
                expected_groups = [[], [], []]
                expected_output_group = 0
                for triplet, is_complete in zip(batched(wfi1 + wfi2 + wfi3 + nfi, 3), groups_are_complete):
                    # We're iterating through the triplets sorted first by observatory, then by time.
                    if not is_complete:
                        # This triplet should be missing a file
                        triplet = list(triplet)
                        triplet.pop(file_to_exclude)
                        input_files.extend(triplet)
                        # We alternate which file to exclude. (The following test handles all permutates of which files
                        # are missing, for a single observatory)
                        file_to_exclude = (file_to_exclude + 1) % 3
                    else:
                        input_files.extend(triplet)
                        expected_groups[expected_output_group].extend(triplet)
                    expected_output_group = (expected_output_group + 1) % 3

                # TODO: we're temporarily excluding NFI in the L2 flows (remove these two lines when that changes)
                input_files = [f for f in input_files if f.observatory != '4']
                expected_groups = [[f for f in g if f.observatory != '4'] for g in expected_groups]

                expected_groups = [set(f.file_id for f in g) for g in expected_groups if len(g)]
                output_groups = group_l2_inputs(input_files)
                output_groups = [set(f.file_id for f in g) for g in output_groups]
                assert len(output_groups) == len(expected_groups)
                for output, expected in zip(output_groups, expected_groups):
                    assert output == expected


def test_group_l2_inputs_single_observatory():
    """
    Ensures that the per-observatory grouping of polarized input images works for any combination of missing inputs
    """
    # First we generate input files---three PZM triplets
    input_files = []
    t0 = datetime(2025, 6, 1, 1)
    for group_dt in [0, 4, 8]:
        for p, dt in zip(['P', 'Z', 'M'], [5, 62, 136]):
            input_files.append(File(level='1', file_type=f"P{p}", observatory='2', file_version='1', software_version='1',
                             date_obs=t0 + timedelta(minutes=group_dt, seconds=dt), state='created', polarization=p))
    for i, file in enumerate(input_files):
        file.file_id = i

    # We'll iterate through the entire possibility grid of missing files. 9 files, so 9 sets of [True, False] (for "is
    # this file included"), and we'll do every combination of choices from each of the 9 sets.
    for files_are_included in itertools.product(*([(True, False)] * 9)):
        selected_files = [f for f, ok in zip(input_files, files_are_included) if ok]
        output_groups = group_l2_inputs_single_observatory(selected_files, expected_sequence=['P', 'Z', 'M'])
        expected_groups = [
            tuple(f for f, ok in zip(input_files[0:3], files_are_included[0:3]) if ok),
            tuple(f for f, ok in zip(input_files[3:6], files_are_included[3:6]) if ok),
            tuple(f for f, ok in zip(input_files[6:9], files_are_included[6:9]) if ok),
        ]
        expected_groups = [group for group in expected_groups if len(group)]
        assert tuple(output_groups) == tuple(expected_groups)


def test_level2_query_ready_files_ignore_missing(db):
    with disable_run_logger():
        with freeze_time(datetime(2023, 1, 2, 0, 0, 0, tzinfo=UTC)) as frozen_datatime:  # noqa: F841
            pipeline_config = {'flows': {'level2': {'ignore_missing_after_days': 1.05}}}
            ready_file_ids = level2_query_ready_files.fn(db, pipeline_config)
            assert len(ready_file_ids) == 0
            pipeline_config = {'flows': {'level2': {'ignore_missing_after_days': 0.95}}}
            ready_file_ids = level2_query_ready_files.fn(db, pipeline_config)
            assert len(ready_file_ids) == 1


def test_level2_query_ready_files_ignore_missing_clear(db):
    with disable_run_logger():
        with freeze_time(datetime(2023, 1, 2, 0, 0, 0, tzinfo=UTC)) as frozen_datatime:  # noqa: F841
            pipeline_config = {'flows': {'level2_clear': {'ignore_missing_after_days': 1.05}}}
            ready_file_ids = level2_query_ready_clear_files.fn(db, pipeline_config)
            assert len(ready_file_ids) == 0
            pipeline_config = {'flows': {'level2_clear': {'ignore_missing_after_days': 0.95}}}
            ready_file_ids = level2_query_ready_clear_files.fn(db, pipeline_config)
            assert len(ready_file_ids) == 1


def test_level2_clear_query_ready_files_unprocessed_L0(db):
    try:
        with disable_run_logger(), freeze_time(datetime(2023, 1, 1, 0, 5, 0)):  # noqa: F841
            pipeline_config = {'flows': {'level2_clear': {'ignore_missing_after_days': 0}}}
            ready_file_ids = level2_query_ready_clear_files.fn(db, pipeline_config)
            assert len(ready_file_ids) == 1

            level0_file = File(level="0",
                               file_type="CR",
                               observatory="2",
                               state="progressed",
                               file_version="none",
                               software_version="none",
                               date_obs=datetime(2023, 1, 1, 0, 0, 0))
            db.add(level0_file)

            ready_file_ids = level2_query_ready_clear_files.fn(db, pipeline_config)
            assert len(ready_file_ids) == 0
    finally:
        db.rollback()


def test_level2_construct_file_info():
    pipeline_config_path = os.path.join(TEST_DIR, "punchpipe_config.yaml")
    pipeline_config = load_pipeline_configuration(pipeline_config_path)

    level1_file = [File(level='1',
                       file_type='PT',
                       observatory='M',
                       state='created',
                       file_version='none',
                       software_version='none',
                       date_obs=datetime.now(UTC))]
    constructed_file_info = level2_construct_file_info(level1_file, pipeline_config)[0]
    assert constructed_file_info.level == '2'
    assert constructed_file_info.file_type == level1_file[0].file_type
    assert constructed_file_info.observatory == level1_file[0].observatory
    assert constructed_file_info.file_version == "0.0.1"
    assert constructed_file_info.software_version == __version__
    assert constructed_file_info.date_obs == level1_file[0].date_obs
    assert constructed_file_info.polarization == 'Y'
    assert constructed_file_info.state == "planned"


def test_level2_construct_flow_info():
    pipeline_config_path = os.path.join(TEST_DIR, "punchpipe_config.yaml")
    pipeline_config = load_pipeline_configuration(pipeline_config_path)
    level1_file = [File(level="1",
                       file_type='XX',
                       observatory='0',
                       state='created',
                       file_version='none',
                       software_version='none',
                       date_obs=datetime.now(UTC))]
    level2_file = level2_construct_file_info(level1_file, pipeline_config)
    flow_info = level2_construct_flow_info(level1_file, level2_file, pipeline_config)

    assert flow_info.flow_type == 'level2'
    assert flow_info.state == "planned"
    assert flow_info.flow_level == "2"
    assert flow_info.priority == 1000


def test_level2_scheduler_flow(db):
    pipeline_config_path = os.path.join(TEST_DIR, "punchpipe_config.yaml")
    with prefect_test_harness():
        level2_scheduler_flow(pipeline_config_path, db)
    results = db.query(Flow).where(Flow.state == 'planned').all()
    assert len(results) == 1


def test_level2_process_flow(db):
    pass
