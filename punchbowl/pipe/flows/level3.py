import json
from datetime import datetime, timedelta

from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from sqlalchemy import and_

from punchbowl import __version__
from punchbowl.level3.flow import level3_core_flow, level3_PIM_flow
from punchbowl.pipe.control.db import File, Flow, get_closest_after_file, get_closest_before_file, get_closest_file
from punchbowl.pipe.control.processor import generic_process_flow_logic
from punchbowl.pipe.control.scheduler import generic_scheduler_flow_logic
from punchbowl.pipe.control.util import get_database_session
from punchbowl.pipe.flows.util import file_name_to_full_path


def get_valid_starfields(session, f: File, timedelta_window: timedelta, file_type: str = "PS"):
    valid_star_start, valid_star_end = f.date_obs - timedelta_window, f.date_obs + timedelta_window
    return (session.query(File).filter(File.state == "created").filter(File.level == "3")
                        .filter(File.file_type == file_type).filter(File.observatory == "M")
                        .filter(and_(f.date_obs >= valid_star_start,
                                     f.date_obs <= valid_star_end)).all())


def get_valid_fcorona_models(session, f: File, before_timedelta: timedelta, after_timedelta: timedelta, file_type="PF"):
    valid_fcorona_start, valid_fcorona_end = f.date_obs - before_timedelta, f.date_obs + after_timedelta
    return (session.query(File).filter(File.state == "created").filter(File.level == "3")
                      .filter(File.file_type == file_type).filter(File.observatory == "M")
                      .filter(File.date_obs >= valid_fcorona_start)
                      .filter(File.date_obs <= valid_fcorona_end).all())


@task(cache_policy=NO_CACHE)
def level3_PTM_query_ready_files(session, pipeline_config: dict, reference_time=None, max_n=9e99):
    logger = get_run_logger()
    all_ready_files = session.query(File).where(and_(and_(File.state.in_(["progressed", "created"]),
                                                          File.level == "2"),
                                                     File.file_type == "PT")).order_by(File.date_obs.asc()).all()
    logger.info(f"{len(all_ready_files)} Level 3 PTM files need to be processed.")

    actually_ready_files = []
    for f in all_ready_files:
        # TODO put magic numbers in config
        valid_starfields = get_valid_starfields(session, f, timedelta_window=timedelta(days=14))

        if len(valid_starfields) >= 1:
            actually_ready_files.append(f)
            if len(actually_ready_files) >= max_n:
                break
    logger.info(f"{len(actually_ready_files)} Level 3 PTM files selected with necessary calibration data.")

    return [[f.file_id] for f in actually_ready_files]


def level3_PTM_construct_flow_info(level2_files: list[File], level3_file: File,
                                   pipeline_config: dict, session=None, reference_time=None):
    session = get_database_session()  # TODO: replace so this works in the tests by passing in a test

    flow_type = "level3_PTM"
    state = "planned"
    creation_time = datetime.now()
    priority = pipeline_config["flows"][flow_type]["priority"]["initial"]

    starfield = get_closest_file(level2_files[0],
                                 get_valid_starfields(session,
                                                      level2_files[0],
                                                      timedelta_window=timedelta(days=14)))
    call_data = json.dumps(
        {
            "data_list": [level2_file.filename() for level2_file in level2_files],
            "starfield_background_path": starfield.filename(),
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


def level3_PTM_construct_file_info(input_files: list[File], pipeline_config: dict, reference_time=None) -> list[File]:
    date_obses = [f.date_obs for f in input_files]

    return [File(
                level="3",
                file_type="PT",
                observatory="M",
                file_version=pipeline_config["file_version"],
                software_version=__version__,
                date_obs=input_files[0].date_obs,
                state="planned",
                date_beg=min(date_obses),
                date_end=max(date_obses),
                outlier=any(file.outlier for file in input_files),
                bad_packets=any(file.bad_packets for file in input_files),
            )]


@flow
def level3_PTM_scheduler_flow(pipeline_config_path=None, session=None, reference_time=None):
    generic_scheduler_flow_logic(
        level3_PTM_query_ready_files,
        level3_PTM_construct_file_info,
        level3_PTM_construct_flow_info,
        pipeline_config_path,
        reference_time=reference_time,
        session=session,
    )


def level3_PTM_call_data_processor(call_data: dict, pipeline_config, session=None) -> dict:
    for key in ["data_list", "before_f_corona_model_path", "after_f_corona_model_path", "starfield_background_path"]:
        call_data[key] = file_name_to_full_path(call_data[key], pipeline_config["root"])
    return call_data


@flow
def level3_PTM_process_flow(flow_id: int | list[int], pipeline_config_path=None, session=None):
    generic_process_flow_logic(flow_id, level3_core_flow, pipeline_config_path, session=session,
                               call_data_processor=level3_PTM_call_data_processor)


@task(cache_policy=NO_CACHE)
def level3_PIM_query_ready_files(session, pipeline_config: dict, reference_time=None, max_n=9e99):
    logger = get_run_logger()
    all_ready_files = session.query(File).where(and_(and_(File.state == "created",
                                                          File.level == "2"),
                                                     File.file_type == "PT")).order_by(File.date_obs.asc()).all()
    logger.info(f"{len(all_ready_files)} Level 3 PTM files need to be processed.")

    actually_ready_files = []
    for f in all_ready_files:
        valid_before_fcorona_models = get_valid_fcorona_models(session, f,
                                                               before_timedelta=timedelta(days=14),
                                                               after_timedelta=timedelta(days=0))
        valid_after_fcorona_models = get_valid_fcorona_models(session, f,
                                                               before_timedelta=timedelta(days=0),
                                                               after_timedelta=timedelta(days=14))
        if len(valid_before_fcorona_models) >= 1 and len(valid_after_fcorona_models) >= 1:
            actually_ready_files.append(f)
            if len(actually_ready_files) >= max_n:
                break
    logger.info(f"{len(actually_ready_files)} Level 2 PTM files selected with necessary calibration data.")

    return [[f.file_id] for f in actually_ready_files]


def level3_PIM_construct_flow_info(level2_files: list[File], level3_file: File, pipeline_config: dict,
                                   session=None, reference_time=None):
    session = get_database_session()  # TODO: replace so this works in the tests by passing in a test

    flow_type = "level3_PIM"
    state = "planned"
    creation_time = datetime.now()
    priority = pipeline_config["flows"][flow_type]["priority"]["initial"]
    before_models = get_valid_fcorona_models(session,
                                             level2_files[0],
                                             before_timedelta=timedelta(days=90),
                                             after_timedelta=timedelta(days=0))
    after_models = get_valid_fcorona_models(session,
                                            level2_files[0],
                                            before_timedelta=timedelta(days=0),
                                            after_timedelta=timedelta(days=90))
    f_corona_before = get_closest_before_file(level2_files[0], before_models)
    f_corona_after = get_closest_after_file(level2_files[0], after_models)
    call_data = json.dumps(
        {
            "data_list": [level2_file.filename() for level2_file in level2_files],
            # TODO put magic numbers in config
            "before_f_corona_model_path": f_corona_before.filename(),
            "after_f_corona_model_path": f_corona_after.filename(),
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


def level3_PIM_construct_file_info(level2_files: list[File], pipeline_config: dict, reference_time=None) -> list[File]:
    date_obses = [f.date_obs for f in level2_files]

    return [File(
                level="3",
                file_type="PI",
                observatory="M",
                file_version=pipeline_config["file_version"],
                software_version=__version__,
                date_obs=level2_files[0].date_obs,
                state="planned",
                date_beg=min(date_obses),
                date_end=max(date_obses),
                outlier=any(file.outlier for file in level2_files),
                bad_packets=any(file.bad_packets for file in level2_files),
            )]


@flow
def level3_PIM_scheduler_flow(pipeline_config_path: str | None = None,
                              session=None,
                              reference_time: datetime | None = None):
    generic_scheduler_flow_logic(
        level3_PIM_query_ready_files,
        level3_PIM_construct_file_info,
        level3_PIM_construct_flow_info,
        pipeline_config_path,
        reference_time=reference_time,
        session=session,
    )


def level3_PIM_call_data_processor(call_data: dict, pipeline_config, session=None) -> dict:
    for key in ["data_list", "before_f_corona_model_path", "after_f_corona_model_path"]:
        call_data[key] = file_name_to_full_path(call_data[key], pipeline_config["root"])
    return call_data


@flow
def level3_PIM_process_flow(flow_id: int | list[int], pipeline_config_path=None, session=None):
    generic_process_flow_logic(flow_id, level3_PIM_flow, pipeline_config_path, session=session,
                               call_data_processor=level3_PIM_call_data_processor)


@task(cache_policy=NO_CACHE)
def level3_CIM_query_ready_files(session, pipeline_config: dict, reference_time=None, max_n=9e99):
    logger = get_run_logger()
    all_ready_files = session.query(File).where(and_(and_(File.state == "created",
                                                          File.level == "2"),
                                                     File.file_type == "CT")).order_by(File.date_obs.asc()).all()
    logger.info(f"{len(all_ready_files)} Level 2 CTM files need to be processed.")

    actually_ready_files = []
    for f in all_ready_files:
        valid_before_fcorona_models = get_valid_fcorona_models(session, f,
                                                               before_timedelta=timedelta(days=14),
                                                               after_timedelta=timedelta(days=0),
                                                               file_type="CF")
        valid_after_fcorona_models = get_valid_fcorona_models(session, f,
                                                              before_timedelta=timedelta(days=0),
                                                              after_timedelta=timedelta(days=14),
                                                              file_type="CF")
        if len(valid_before_fcorona_models) >= 1 and len(valid_after_fcorona_models) >= 1:
            actually_ready_files.append(f)
            if len(actually_ready_files) >= max_n:
                break
    logger.info(f"{len(actually_ready_files)} Level 2 CTM files selected with necessary calibration data.")

    return [[f.file_id] for f in actually_ready_files]


def level3_CIM_construct_flow_info(level2_files: list[File], level3_file: File, pipeline_config: dict,
                                   session=None, reference_time=None):
    session = get_database_session()  # TODO: replace so this works in the tests by passing in a test

    flow_type = "level3_CIM"
    state = "planned"
    creation_time = datetime.now()
    priority = pipeline_config["flows"][flow_type]["priority"]["initial"]
    before_models = get_valid_fcorona_models(session,
                                             level2_files[0],
                                             before_timedelta=timedelta(days=90),
                                             after_timedelta=timedelta(days=0),
                                             file_type="CF")
    after_models = get_valid_fcorona_models(session,
                                            level2_files[0],
                                            before_timedelta=timedelta(days=0),
                                            after_timedelta=timedelta(days=90),
                                            file_type="CF")
    f_corona_before = get_closest_before_file(level2_files[0], before_models)
    f_corona_after = get_closest_after_file(level2_files[0], after_models)
    call_data = json.dumps(
        {
            "data_list": [level2_file.filename() for level2_file in level2_files],
            "before_f_corona_model_path": f_corona_before.filename(),
            "after_f_corona_model_path": f_corona_after.filename(),
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


def level3_CIM_construct_file_info(level2_files: list[File], pipeline_config: dict, reference_time=None) -> list[File]:
    date_obses = [f.date_obs for f in level2_files]

    return [File(
                level="3",
                file_type="CI",
                observatory="M",
                file_version=pipeline_config["file_version"],
                software_version=__version__,
                date_obs=level2_files[0].date_obs,
                state="planned",
                date_beg=min(date_obses),
                date_end=max(date_obses),
                outlier=any(file.outlier for file in level2_files),
                bad_packets=any(file.bad_packets for file in level2_files),
            )]


@flow
def level3_CIM_scheduler_flow(pipeline_config_path: str | None = None,
                              session=None,
                              reference_time: datetime | None = None):
    generic_scheduler_flow_logic(
        level3_CIM_query_ready_files,
        level3_CIM_construct_file_info,
        level3_CIM_construct_flow_info,
        pipeline_config_path,
        reference_time=reference_time,
        session=session,
    )


def level3_CIM_call_data_processor(call_data: dict, pipeline_config, session=None) -> dict:
    for key in ["data_list", "before_f_corona_model_path", "after_f_corona_model_path"]:
        call_data[key] = file_name_to_full_path(call_data[key], pipeline_config["root"])
    return call_data


@flow
def level3_CIM_process_flow(flow_id: int | list[int], pipeline_config_path=None, session=None):
    # NOTE: this is not a typo... we're using the PIM core flow for this because it's flexible
    generic_process_flow_logic(flow_id, level3_PIM_flow, pipeline_config_path, session=session,
                               call_data_processor=level3_CIM_call_data_processor)


@task(cache_policy=NO_CACHE)
def level3_CTM_query_ready_files(session, pipeline_config: dict, reference_time=None, max_n=9e99):
    logger = get_run_logger()
    all_ready_files = session.query(File).where(and_(and_(File.state.in_(["created"]),
                                                          File.level == "3"),
                                                     File.file_type == "CI")).order_by(File.date_obs.asc()).all()
    logger.info(f"{len(all_ready_files)} Level 3 CIM files need to be processed.")

    actually_ready_files = []
    for f in all_ready_files:
        # # TODO put magic numbers in config
        valid_starfields = get_valid_starfields(session, f, timedelta_window=timedelta(days=14), file_type="CS")

        if len(valid_starfields) >= 1:
            actually_ready_files.append(f)
            if len(actually_ready_files) >= max_n:
                break
    logger.info(f"{len(actually_ready_files)} Level 3 CIM files selected with necessary calibration data.")

    return [[f.file_id] for f in actually_ready_files]


def level3_CTM_construct_flow_info(level2_files: list[File], level3_file: File,
                                   pipeline_config: dict, session=None, reference_time=None):
    session = get_database_session()  # TODO: replace so this works in the tests by passing in a test

    flow_type = "level3_CTM"
    state = "planned"
    creation_time = datetime.now()
    priority = pipeline_config["flows"][flow_type]["priority"]["initial"]

    starfield = get_closest_file(level2_files[0],
                                 get_valid_starfields(session,
                                                      level2_files[0],
                                                      timedelta_window=timedelta(days=90),
                                                      file_type="CS"))
    call_data = json.dumps(
        {
            "data_list": [level2_file.filename() for level2_file in level2_files],
            "starfield_background_path": starfield.filename(),
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


def level3_CTM_construct_file_info(input_files: list[File], pipeline_config: dict, reference_time=None ) -> list[File]:
    date_obses = [f.date_obs for f in input_files]

    return [File(
                level="3",
                file_type="CT",
                observatory="M",
                file_version=pipeline_config["file_version"],
                software_version=__version__,
                date_obs=input_files[0].date_obs,
                state="planned",
                date_beg=min(date_obses),
                date_end=max(date_obses),
                outlier=any(file.outlier for file in input_files),
                bad_packets=any(file.bad_packets for file in input_files),
            )]


@flow
def level3_CTM_scheduler_flow(pipeline_config_path=None, session=None, reference_time=None):
    generic_scheduler_flow_logic(
        level3_CTM_query_ready_files,
        level3_CTM_construct_file_info,
        level3_CTM_construct_flow_info,
        pipeline_config_path,
        reference_time=reference_time,
        session=session,
    )


def level3_CTM_call_data_processor(call_data: dict, pipeline_config, session=None) -> dict:
    for key in ["data_list" , "starfield_background_path"]:
        call_data[key] = file_name_to_full_path(call_data[key], pipeline_config["root"])
    return call_data


@flow
def level3_CTM_process_flow(flow_id: int | list[int], pipeline_config_path=None, session=None):
    generic_process_flow_logic(flow_id, level3_core_flow, pipeline_config_path, session=session,
                               call_data_processor=level3_CTM_call_data_processor)

import os
from datetime import UTC, datetime

from prefect import flow, task
from punchpipe import __version__
from punchpipe.control.db import File, Flow
from punchpipe.control.processor import generic_process_flow_logic
from punchpipe.control.scheduler import generic_scheduler_flow_logic

from punchbowl.level3.flow import generate_level3_low_noise_flow

# TODO - repeat below logic for the final PAM set

@task
def level3_CAM_query_ready_files(session, pipeline_config: dict, reference_time=None, max_n=9e99):
    logger = get_run_logger()
    all_ready_files = session.query(File).where(and_(and_(File.state.in_(["progressed", "created"]),
                                                          File.level == "3"),
                                                     File.file_type == "CT")).order_by(File.date_obs.asc()).all()
    # TODO - need to grab data from sets of rotation. look at movie processor for inspiration
    logger.info(f"{len(all_ready_files)} Level 3 CTM files need to be processed to low-noise.")

    actually_ready_files = []
    for f in all_ready_files:
        # TODO - remove this unless the time check happens here
        actually_ready_files.append(f)
        if len(actually_ready_files) >= max_n:
            break
    logger.info(f"{len(actually_ready_files)} Level 3 CTM files selected for low-noise processing")

    return [[f.file_id] for f in actually_ready_files]


@task
def level3_CAM_construct_flow_info(level3_files: list[File], level3_file_out: File,
                                   pipeline_config: dict, session=None, reference_time=None):

    flow_type = "level3_CAM"
    state = "planned"
    creation_time = datetime.now(UTC)
    priority = pipeline_config["flows"][flow_type]["priority"]["initial"]

    call_data = json.dumps(
        {
            "data_list": [
                os.path.join(level3_file.directory(pipeline_config["root"]), level3_file.filename())
                for level3_file in level3_files
            ],
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
def level3_CAM_construct_file_info(level3_files: list[File], pipeline_config: dict, reference_time=None) -> list[File]:
    return [File(
                level="3",
                file_type="CA",
                observatory="M",
                file_version=pipeline_config["file_version"],
                software_version=__version__,
                date_obs=level3_files[0].date_obs, # TODO - set to date avg
                state="planned",
            )]


@flow
def level3_CAM_scheduler_flow(pipeline_config_path=None, session=None, reference_time=None):
    generic_scheduler_flow_logic(
        level3_CAM_query_ready_files,
        level3_CAM_construct_file_info,
        level3_CAM_construct_flow_info,
        pipeline_config_path,
        reference_time=reference_time,
        session=session,
    )


@flow
def level3_CAM_process_flow(flow_id: int, pipeline_config_path=None, session=None):
    generic_process_flow_logic(flow_id, generate_level3_low_noise_flow, pipeline_config_path, session=session)
