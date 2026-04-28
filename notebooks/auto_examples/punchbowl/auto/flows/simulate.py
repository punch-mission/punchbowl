"""Run the entire pipeline backward."""
import os
import glob
import json
from datetime import UTC, datetime
from collections.abc import Callable

from dateutil.parser import parse as parse_datetime_str
from prefect import flow, get_run_logger
from prefect.context import get_run_context
from simpunch.flow import generate_flow

from punchbowl.auto.control import cache_layer
from punchbowl.auto.control.db import File, Flow
from punchbowl.auto.control.util import get_database_session, load_pipeline_configuration


@flow
def simpunch_scheduler_flow(pipeline_config_path=None, session=None, reference_time: datetime | str | None | list = None):
    pipeline_config = load_pipeline_configuration(pipeline_config_path)
    flow_type = "simpunch"
    state = "planned"
    creation_time = datetime.now()
    priority = pipeline_config["flows"][flow_type]["priority"]["initial"]

    if session is None:
        session = get_database_session()

    if not isinstance(reference_time, list):
        reference_time = [reference_time]
    for ref_time in reference_time:
        call_data = json.dumps(
            {
                "date_obs": ref_time or str(datetime.now(UTC)),
                "simulation_start": pipeline_config["flows"][flow_type]["options"].get("simulation_start", ""),
                "simulation_cadence_minutes": pipeline_config["flows"][flow_type]["options"].get("simulation_cadence_minutes", 4.0),
                "gamera_files_dir": pipeline_config["flows"][flow_type]["options"].get("gamera_files_dir", ""),
                "out_dir": pipeline_config["flows"][flow_type]["options"].get("out_dir", ""),
                "backward_psf_model_path": pipeline_config["flows"][flow_type]["options"].get("backward_psf_model_path", ""),
                "wfi_quartic_backward_model_path": pipeline_config["flows"][flow_type]["options"].get("wfi_quartic_backward_model_path", ""),
                "nfi_quartic_backward_model_path": pipeline_config["flows"][flow_type]["options"].get("nfi_quartic_backward_model_path", ""),
                "transient_probability": pipeline_config["flows"][flow_type]["options"].get("transient_probability", 0),
                "shift_pointing": pipeline_config["flows"][flow_type]["options"].get("shift_pointing", False),
            },
        )
        new_flow = Flow(
            flow_type=flow_type,
            flow_level="S",
            state=state,
            creation_time=creation_time,
            priority=priority,
            call_data=call_data,
        )

        session.add(new_flow)
    session.commit()

@flow
def simpunch_core_flow(
        date_obs: datetime | str,
        simulation_start: datetime | str,
        simulation_cadence_minutes: float,
        gamera_files_dir: str,
        out_dir: str,
        backward_psf_model_path: str | Callable,
        wfi_quartic_backward_model_path: str | Callable,
        nfi_quartic_backward_model_path: str | Callable,
        transient_probability: float = 0.03,
        shift_pointing: bool = False) -> list[str]:

    logger = get_run_logger()

    if isinstance(date_obs, str):
        date_obs = parse_datetime_str(date_obs).replace(tzinfo=UTC)
    if isinstance(simulation_start, str):
        simulation_start = parse_datetime_str(simulation_start).replace(tzinfo=UTC)

    logger.info(f"Running for {date_obs}")

    tb_files = sorted(glob.glob(gamera_files_dir + "/*_TB.fits"))
    pb_files = sorted(glob.glob(gamera_files_dir + "/*_PB.fits"))
    simulation_frame_count = len(tb_files)

    minutes_after_start = (date_obs - simulation_start).total_seconds() / 60
    raw_index = minutes_after_start / simulation_cadence_minutes
    index = int(raw_index % simulation_frame_count)
    logger.info(f"Running on index {index}")

    file_tb = tb_files[index]
    file_pb = pb_files[index]
    logger.info(f"file_tb = {file_tb}")
    logger.info(f"file_pb = {file_pb}")

    return generate_flow(file_tb, file_pb, date_obs, out_dir, backward_psf_model_path,wfi_quartic_backward_model_path,
                  nfi_quartic_backward_model_path, transient_probability, shift_pointing)



@flow
def simpunch_process_flow(flow_id: int, pipeline_config_path=None, session=None):
    logger = get_run_logger()

    if session is None:
        session = get_database_session()

    # fetch the appropriate flow db entry
    flow_db_entry = session.query(Flow).where(Flow.flow_id == flow_id).one()
    logger.info(f"Running on flow db entry with id={flow_db_entry.flow_id}.")

    # update the processing flow name with the flow run name from Prefect
    flow_run_context = get_run_context()
    flow_db_entry.flow_run_name = flow_run_context.flow_run.name
    flow_db_entry.flow_run_id = flow_run_context.flow_run.id
    flow_db_entry.state = "running"
    flow_db_entry.start_time = datetime.now()
    session.commit()

    # load the call data and launch the core flow
    flow_call_data = json.loads(flow_db_entry.call_data)
    logger.info(f"Running with {flow_call_data}")
    flow_call_data["backward_psf_model_path"] = cache_layer.psf.wrap_if_appropriate(
        flow_call_data["backward_psf_model_path"])
    flow_call_data["wfi_quartic_backward_model_path"] = cache_layer.quartic_coefficients.wrap_if_appropriate(
        flow_call_data["wfi_quartic_backward_model_path"])
    flow_call_data["nfi_quartic_backward_model_path"] = cache_layer.quartic_coefficients.wrap_if_appropriate(
        flow_call_data["nfi_quartic_backward_model_path"])
    try:
        out_filenames = simpunch_core_flow(**flow_call_data)
    except Exception as e:
        flow_db_entry.state = "failed"
        flow_db_entry.end_time = datetime.now()
        session.commit()
        raise e
    else:
        file_db_entries = []
        for filename in out_filenames:
            base_filename = os.path.basename(filename)
            if base_filename.startswith("PUNCH_L0"):
                file_db_entries.append(File(
                    level="0",
                    file_type=base_filename[9:11],
                    observatory=base_filename[11],
                    file_version=base_filename[:-5].split("_")[-1][1:],
                    software_version="synth",
                    date_created=datetime.now(),
                    date_obs=datetime.strptime(base_filename[13:27], "%Y%m%d%H%M%S"),
                    state="created",
                ))
        session.add_all(file_db_entries)

        flow_db_entry.state = "completed"
        flow_db_entry.end_time = datetime.now()
        # Note: the file_db_entry gets updated above in the writing step because it could be created or blank
        session.commit()
