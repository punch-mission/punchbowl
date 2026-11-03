import json
import inspect
from bisect import bisect_left
from datetime import time, datetime, timedelta

import numpy as np
from dateutil.parser import parse as parse_datetime_str
from prefect import flow, task

from punchbowl import __version__
from punchbowl.auto.control.db import File, Flow
from punchbowl.auto.control.processor import generic_process_flow_logic
from punchbowl.auto.control.scheduler import generic_scheduler_flow_logic
from punchbowl.auto.control.util import get_database_session, load_pipeline_configuration
from punchbowl.auto.flows.util import file_name_to_full_path
from punchbowl.level3.flow import generate_level3_velocity_flow
from punchbowl.level3.velocity import get_cadence_min, get_frame_slots, select_frame_slots, track_velocity
from punchbowl.prefect import get_logger

# Defaults of the flow tracking parameters, taken from track_velocity which is authoritative for them. They are used
# for any parameter not set in the pipeline configuration, so call_data always holds the full set of parameters.
TRACK_VELOCITY_DEFAULTS = {name: param.default
                           for name, param in inspect.signature(track_velocity).parameters.items()
                           if param.default is not inspect.Parameter.empty and name != "reference_time"}
DEFAULT_PRODUCT_CODE = "PTM"
# Velocity maps are made at the frames nearest to 03:00, 09:00, 15:00 and 21:00 UTC every day
NOMINAL_REFERENCE_HOUR = 3
NOMINAL_REFERENCE_CADENCE = timedelta(hours=6)


class VAMInputGroup(list):
    """Input files of one velocity map, with the authoritative reference time of that map.

    Consecutive velocity maps have overlapping input windows, so the same File object can be in two groups.
    Storing the reference time on the files (as other flows do with ``_reference_time``) would let one group
    overwrite the other's value, so it is stored on the group instead.
    """

    def __init__(self, files: list[File], reference_time: datetime):
        super().__init__(files)
        self.reference_time = reference_time


def get_flow_parameter(flow_config: dict, key: str) -> int | float | bool | list[int | float]:
    """Get a flow tracking parameter from the pipeline configuration, defaulting to that of track_velocity.
    The real pipeline config is an external YAML file currently not exposed to this codebase,
    so type hints are only guesses.

    Returns a scalar (int, float or bool) or, for sequence parameters such as annuli_centers_rs, a list of numbers,
    so that it can be serialized in the flow call data.
    """
    value = flow_config.get(key, TRACK_VELOCITY_DEFAULTS[key])
    # JSON-serializable, e.g. for the default tuple of annuli centers
    return list(value) if isinstance(value, (tuple, np.ndarray)) else value


def get_product_code(flow_config: dict) -> str:
    """Get the product code of the files to use as input for velocity tracking from the pipeline configuration."""
    product_code = flow_config.get("product_code", DEFAULT_PRODUCT_CODE)
    # Raises a ValueError for products unsupported by flow tracking
    get_cadence_min(product_code)
    return product_code


def get_nominal_reference_time(obs_time: datetime, anchor: datetime) -> datetime:
    """Get the nominal reference time (03:00, 09:00, 15:00 or 21:00 UTC) nearest to a time."""
    return anchor + round((obs_time - anchor) / NOMINAL_REFERENCE_CADENCE) * NOMINAL_REFERENCE_CADENCE


@task
def level3_VAM_query_ready_files(session, pipeline_config: dict, reference_time: datetime=None, max_n: float=100) -> list:
    """Queries files ready for velocity tracking.

    Velocity maps are made at the nominal reference times 03:00, 09:00, 15:00 and 21:00 UTC since t0, on a 6-hour
    grid anchored at 03:00 UTC on the day of t0. The authoritative reference time of each map is the DATE-OBS of
    the input frame nearest to its nominal time.
    Each map uses the input frames centered on its reference time that cover the flow map integration window and
    the temporal-average buffer required by track_velocity. Missing frames are tolerated within the limits of
    punchbowl.level3.velocity.check_missing_runs, and are interpolated by track_velocity. Consecutive windows overlap.
    Since t0 is the time of the first data, the maps at the first nominal times after t0 lack frames for their
    temporal-average buffer and flow window, and are skipped.

    Parameters
    ----------
    session
        Database session
    pipeline_config : dict
        Pipeline configuration dictionary
    max_n : float, optional
        Max number of files to specify ready, by default 100

    Returns
    -------
    list
        Groups of ready files, as VAMInputGroup carrying their reference time, most recent first
    """
    flow_type = "level3_VAM"
    logger = get_logger()
    flow_config = pipeline_config["flows"][flow_type]
    product_code = get_product_code(flow_config)
    window_params = {key: get_flow_parameter(flow_config, key)
                     for key in ("flow_window_hours", "time_avg_hours")}

    t0 = parse_datetime_str(flow_config["t0"])
    # The nominal reference times are anchored at 03:00 UTC on the day of the first data
    anchor = datetime.combine(t0.date(), time(NOMINAL_REFERENCE_HOUR))

    all_ready_files = (session.query(File)
                       .filter(File.state.in_(["created", "progressed"]))
                       .filter(File.level == "3")
                       .filter(File.file_type == product_code[:2])
                       .filter(File.observatory == product_code[2])
                       .filter(File.date_obs >= t0)
                       .order_by(File.date_obs.asc()).all())

    if len(all_ready_files) == 0:
        return []

    # Nominal reference times for which a velocity map already exists (presumably incomplete).
    # TODO: it would be better to regenerate the file, but we don't have a way to do that sensibly now
    existing_nominal_times = {get_nominal_reference_time(date_obs, anchor)
                              for (date_obs,) in session.query(File.date_obs)
                              .filter(File.level == "3")
                              .filter(File.file_type == "VA")
                              .filter(File.observatory == "M")
                              .filter(File.date_obs >= t0)}

    obs_times = [f.date_obs for f in all_ready_files]
    slots, slot_times = get_frame_slots(obs_times, get_cadence_min(product_code))

    # Work backwards from the latest nominal reference time toward t0
    nominal_time = get_nominal_reference_time(obs_times[-1], anchor)
    if nominal_time > obs_times[-1]:
        nominal_time -= NOMINAL_REFERENCE_CADENCE

    grouped_ready_files = []
    while nominal_time >= t0 and len(grouped_ready_files) < max_n:
        this_nominal_time = nominal_time
        nominal_time -= NOMINAL_REFERENCE_CADENCE
        if this_nominal_time in existing_nominal_times:
            continue
        # The reference time is the observation time of the frame nearest to the nominal time
        i = bisect_left(obs_times, this_nominal_time)
        nearest = min((j for j in (i - 1, i) if 0 <= j < len(obs_times)),
                      key=lambda j: abs(obs_times[j] - this_nominal_time))
        ref_time = obs_times[nearest]
        try:
            first, last, _, _ = select_frame_slots(slots, slot_times, product_code,
                                                   reference_time=ref_time, **window_params)
        except ValueError as e:
            # Not enough frames yet, or too many missing frames
            logger.info(f"Skipping velocity map at {this_nominal_time.isoformat()}: {e}")
            continue
        group = [all_ready_files[slot] for slot in slots[first:last + 1] if slot is not None]
        grouped_ready_files.append(VAMInputGroup(group, ref_time))

    logger.info(f"{len(grouped_ready_files)} groups heading out")
    return grouped_ready_files


@task
def level3_VAM_construct_flow_info(level3_input_files: VAMInputGroup,
                                   level3_velocity_file: File,
                                   pipeline_config: dict,
                                   reference_time: datetime | None = None,
                                   session=None) -> Flow:
    """Constructs velocity tracking flow

    Parameters
    ----------
    level3_input_files : VAMInputGroup
        Input files used for velocity tracking, with the reference time of the velocity map
    level3_velocity_file : File
        Output file path for measurement
    pipeline_config : dict
        Pipeline configuration settings
    reference_time : datetime, optional
        Unused: the reference time is the one set on the group of input files by level3_VAM_query_ready_files
    session : _type_, optional
        Database session, by default None

    Returns
    -------
    Flow
        Velocity tracking flow to run
    """
    flow_type = "level3_VAM"
    state = "planned"
    creation_time = datetime.now()
    flow_config = pipeline_config["flows"][flow_type]
    priority = flow_config.get("priority", {}).get("initial", 1)
    reference_time = level3_input_files.reference_time
    # Any parameter not set in the pipeline config falls back to the authoritative defaults of track_velocity.
    call_data = json.dumps(
        {
            "files": [input_file.filename() for input_file in level3_input_files],
            "product_code": get_product_code(flow_config),
            # Full precision, so that the reference time is exactly the DATE-OBS of its frame
            "reference_time": reference_time.isoformat(),
            **{key: get_flow_parameter(flow_config, key) for key in TRACK_VELOCITY_DEFAULTS},
        },
    )
    return Flow(
        flow_type=flow_type,
        state=state,
        flow_level="3",
        creation_time=creation_time,
        priority=priority,
        call_data=call_data,
    )

@task
def level3_VAM_construct_file_info(level3_files: VAMInputGroup, pipeline_config: dict,
                                   reference_time: datetime | None = None) -> list[File]:
    """Constructs specified file information

    Parameters
    ----------
    level3_files : VAMInputGroup
        Input files used for velocity tracking, with the reference time of the velocity map
    pipeline_config : dict
        Pipeline configuration settings
    reference_time : datetime, optional
        Unused: the reference time is the one set on the group of input files by level3_VAM_query_ready_files

    Returns
    -------
    list[File]
        List of specified output velocity map files
    """
    return [File(
        level="3",
        file_type="VA",
        observatory="M",
        file_version=pipeline_config["file_version"],
        software_version=__version__,
        date_obs=level3_files.reference_time,
        date_beg=min([f.date_obs for f in level3_files]),
        date_end=max([f.date_obs for f in level3_files]),
        state="planned",
        polarization="Y",
        # Outlier images are excluded from VAMs
        outlier=0,
        bad_packets=False,
    )]


@flow
def level3_VAM_scheduler_flow(pipeline_config_path=None, session=None) -> None:
    """Define the velocity tracking scheduler flow

    Parameters
    ----------
    pipeline_config_path : _type_, optional
        Path to pipeline configuration settings, by default None
    session : _type_, optional
        Database session, by default None
    """
    generic_scheduler_flow_logic(
        level3_VAM_query_ready_files,
        level3_VAM_construct_file_info,
        level3_VAM_construct_flow_info,
        pipeline_config_path,
        update_input_file_state=False,
        session=session,
    )


def level3_VAM_call_data_processor(call_data: dict, pipeline_config, session=None) -> dict:
    """Generate velocity tracking call data

    Parameters
    ----------
    call_data : dict
        Call data
    pipeline_config : dict
        Pipeline configuration settings
    session : _type_, optional
        Database session, by default None

    Returns
    -------
    dict
        Call data
    """
    call_data["files"] = file_name_to_full_path(call_data["files"], pipeline_config["root"])
    return call_data


@flow
def level3_VAM_process_flow(flow_id: int, pipeline_config_path: str = None, session = None) -> None:
    """Define the velocity tracking process flow

    Parameters
    ----------
    flow_id : int
        Flow identification number
    pipeline_config_path : str, optional
        Path to pipeline configuration settings, by default None
    session : optional
        Database session, by default None
    """
    generic_process_flow_logic(flow_id,
                               generate_level3_velocity_flow,
                               pipeline_config_path,
                               session=session,
                               call_data_processor=level3_VAM_call_data_processor,
                               )
