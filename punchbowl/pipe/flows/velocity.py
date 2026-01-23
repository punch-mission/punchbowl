import json
from datetime import UTC, datetime, timedelta

from prefect import flow, get_run_logger, task

from punchbowl import __version__
from punchbowl.level3.velocity import track_velocity
from punchbowl.pipe.control.db import File, Flow
from punchbowl.pipe.control.processor import generic_process_flow_logic
from punchbowl.pipe.control.scheduler import generic_scheduler_flow_logic
from punchbowl.pipe.flows.util import file_name_to_full_path


@task
def level3_vam_query_ready_files(session, pipeline_config: dict, reference_time: datetime):
    logger = get_run_logger()
    start_time = reference_time - timedelta(hours=6)

    logger.info(f"Looking for Level 3 CTM files after {start_time}.")
    ready_files = (session.query(File)
                   .filter(File.state == "created")
                   .filter(File.date_obs >= start_time)
                   .filter(File.level == "3")
                   .filter(File.file_type == "CT")
                   .filter(File.observatory == "M").all())
    logger.info(f"{len(ready_files)} CTM files found for the velocity tracking.")
    return [[f.file_id for f in ready_files]]

@task
def level3_vam_construct_flow_info(level3_ctm_files: list[File],
                                        level3_velocity_file: File,
                                        pipeline_config: dict,
                                        reference_time: datetime,
                                        session=None):
    flow_type = "L3_VAM"
    state = "planned"
    creation_time = datetime.now()
    priority = pipeline_config["flows"][flow_type]["priority"]["initial"]
    call_data = json.dumps(
        {
            "files": [ctm_file.filename() for ctm_file in level3_ctm_files],
            "reference_time": str(reference_time),
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
def level3_vam_construct_file_info(level3_files: list[File], pipeline_config: dict,
                                            reference_time: datetime):
    return [File(
        level="3",
        file_type="VA",
        observatory="M",
        file_version=pipeline_config["file_version"],
        software_version=__version__,
        date_obs=reference_time,
        state="planned",
    )]


@flow
def level3_vam_scheduler_flow(pipeline_config_path=None, session=None, reference_time: datetime | None = None):
    reference_time = reference_time or datetime.now(UTC)

    generic_scheduler_flow_logic(
        level3_vam_query_ready_files,
        level3_vam_construct_file_info,
        level3_vam_construct_flow_info,
        pipeline_config_path,
        update_input_file_state=False,
        reference_time=reference_time,
        session=session,
    )


def level3_vam_call_data_processor(call_data: dict, pipeline_config, session=None) -> dict:
    call_data["files"] = file_name_to_full_path(call_data["files"], pipeline_config["root"])
    return call_data


@flow
def level3_vam_process_flow(flow_id: int, pipeline_config_path=None, session=None):
    generic_process_flow_logic(flow_id,
                               track_velocity,
                               pipeline_config_path,
                               session=session,
                               call_data_processor=level3_vam_call_data_processor,
                               )

@task
def level3_van_query_ready_files(session, pipeline_config: dict, reference_time: datetime):
    logger = get_run_logger()
    start_time = reference_time - timedelta(hours=6)

    logger.info(f"Looking for Level 3 CNN files after {start_time}.")
    ready_files = (session.query(File)
                   .filter(File.state == "created")
                   .filter(File.date_obs >= start_time)
                   .filter(File.level == "3")
                   .filter(File.file_type == "CN")
                   .filter(File.observatory == "N").all())
    logger.info(f"{len(ready_files)} CNN files found for the velocity tracking.")
    return [[f.file_id for f in ready_files]]

@task
def level3_van_construct_flow_info(level3_cnn_files: list[File],
                                        level3_velocity_file: File,
                                        pipeline_config: dict,
                                        reference_time: datetime,
                                        session=None):
    flow_type = "L3_VAN"
    state = "planned"
    creation_time = datetime.now()
    priority = pipeline_config["flows"][flow_type]["priority"]["initial"]
    call_data = json.dumps(
        {
            "files": [cnn_file.filename() for cnn_file in level3_cnn_files],
            "reference_time": str(reference_time),
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
def level3_van_construct_file_info(level3_files: list[File], pipeline_config: dict,
                                            reference_time: datetime):
    return [File(
        level="3",
        file_type="VA",
        observatory="N",
        file_version=pipeline_config["file_version"],
        software_version=__version__,
        date_obs=reference_time,
        state="planned",
    )]


@flow
def level3_van_scheduler_flow(pipeline_config_path=None, session=None, reference_time: datetime | None = None):
    reference_time = reference_time or datetime.now(UTC)

    generic_scheduler_flow_logic(
        level3_van_query_ready_files,
        level3_van_construct_file_info,
        level3_van_construct_flow_info,
        pipeline_config_path,
        update_input_file_state=False,
        reference_time=reference_time,
        session=session,
    )


def level3_van_call_data_processor(call_data: dict, pipeline_config, session=None) -> dict:
    call_data["files"] = file_name_to_full_path(call_data["files"], pipeline_config["root"])
    return call_data


@flow
def level3_van_process_flow(flow_id: int, pipeline_config_path=None, session=None):
    generic_process_flow_logic(flow_id,
                               track_velocity,
                               pipeline_config_path,
                               session=session,
                               call_data_processor=level3_van_call_data_processor,
                               )
