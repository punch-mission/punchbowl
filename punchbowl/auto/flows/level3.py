import os
import json
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from collections import defaultdict

from dateutil.parser import parse as parse_datetime_str
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from sqlalchemy import and_

from punchbowl import __version__
from punchbowl.auto.control.db import File, Flow, get_closest_after_file, get_closest_before_file, get_closest_file
from punchbowl.auto.control.processor import generic_process_flow_logic
from punchbowl.auto.control.scheduler import generic_scheduler_flow_logic
from punchbowl.auto.control.util import get_database_session, group_files_by_time
from punchbowl.auto.flows.util import file_name_to_full_path
from punchbowl.level3.flow import generate_level3_low_noise_flow, level3_core_flow, level3_PIM_CIM_flow
from punchbowl.util import average_datetime


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


def get_fcorona_pairs(session, files: list[File], model_type="PF"):
    # Get all models
    models = (session.query(File)
              .filter(File.file_type == model_type)
              .order_by(File.date_obs.asc()).all())
    models_by_obs = defaultdict(list)
    for model in models:
        models_by_obs[model.observatory].append(model)

    results = []
    for file in files:
        models = models_by_obs[file.observatory]
        for before_model, after_model in pairwise(models):
            # All the models are sorted by date_obs, so there will be exactly one pair where the first is before our
            # file to be calibrated and the second is after
            if before_model.date_obs < file.date_obs < after_model.date_obs:
                break
        else:
            # We didn't find an appropriate pair, so we must still be waiting for the scheduler to fill in here and
            # tell us what's what
            results.append((None, None))
            continue

        if before_model.state == "created" and after_model.state == "created":
            # Good to go!
            results.append((before_model, after_model))
        elif before_model.state == "impossible" or after_model.state == "impossible":
            # Flexible mode---since we'll never be able to generate the "intended" models for this file, let's go for
            # the two closest possible models
            models = [m for m in models if m.state != "impossible"]
            models = sorted(models, key=lambda m: abs(m.date_obs - file.date_obs))
            before_model, after_model = models[:2]
            if after_model.date_obs < before_model.date_obs:
                before_model, after_model = after_model, before_model

            if before_model.state == "created" and after_model.state == "created":
                # Good to go!
                results.append((before_model, after_model))
            else:
                # Wait for files to generate
                results.append((None, None))
        else:
            # If we're here, we're waiting for at least one model to generate, but we do expect it to do so eventually
            results.append((None, None))
    return results


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
    input_file = input_files[0]

    return [File(
                level="3",
                file_type="PT",
                observatory="M",
                file_version=pipeline_config["file_version"],
                software_version=__version__,
                date_obs=input_file.date_obs,
                state="planned",
                date_beg=input_file.date_beg,
                date_end=input_file.date_end,
                outlier=input_file.outlier,
                bad_packets=input_file.bad_packets,
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
    logger.info(f"{len(all_ready_files)} Level 2 PTM files need to be processed.")

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
    input_file = level2_files[0]

    return [File(
                level="3",
                file_type="PI",
                observatory="M",
                file_version=pipeline_config["file_version"],
                software_version=__version__,
                date_obs=input_file.date_obs,
                state="planned",
                date_beg=input_file.date_beg,
                date_end=input_file.date_end,
                outlier=input_file.outlier,
                bad_packets=input_file.bad_packets,
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
    generic_process_flow_logic(flow_id, level3_PIM_CIM_flow, pipeline_config_path, session=session,
                               call_data_processor=level3_PIM_call_data_processor)


@task(cache_policy=NO_CACHE)
def level3_CIM_query_ready_files(session, pipeline_config: dict, reference_time=None, max_n=9e99):
    logger = get_run_logger()
    all_ready_files = (session.query(File).filter(File.state == "created")
                       .filter(File.level == "2")
                       # TODO: This line temporarily excludes NFI
                       .filter(File.observatory.in_(["1", "2", "3"]))
                       .filter(File.file_type == 'XR')
                       # The ascending sort order is expected by the file grouping code
                       .order_by(File.date_obs.asc()).all())
    logger.info(f"{len(all_ready_files)} ready files")
    logger.info(f"{len(all_ready_files)} Level 2 XR files need to be processed.")

    if len(all_ready_files) == 0:
        return []

    grouped_files = group_files_by_time(all_ready_files, max_duration_seconds=10)

    # Add a slight delay to make sure that we're not scooping up files that are being actively
    # written out. If we can count on X-file sets to be completely written out, we can skip the
    # "should we make an incomplete trefoil" logic, deferring that to the L2 scheduler.
    grouped_files = [
        group for group in grouped_files if max(f.date_created for f in group) < datetime.now() - timedelta(minutes=2)]

    target_date = pipeline_config.get("target_date")
    target_date = parse_datetime_str(target_date) if target_date else None
    if target_date:
        # Sort by closeness to the target date
        grouped_files.sort(key=lambda group: abs((group[0].date_obs - target_date).total_seconds()))
    else:
        # Switch to most-recent-first order
        grouped_files = grouped_files[::-1]

    all_files = [f for group in grouped_files for f in group]

    fcorona_models = get_fcorona_pairs(session, all_files, model_type="CF")
    file_to_fcor_model = {f.file_id: m for f, m in zip(all_files, fcorona_models)}

    actually_ready_groups = []
    missing_fcor = []
    for group in grouped_files:
        all_set = True
        for f in group:
            if file_to_fcor_model[f.file_id] == (None, None):
                missing_fcor.append(f)
                all_set = False
            else:
                f.fcor_models = file_to_fcor_model[f.file_id]
        if not all_set:
            continue
        actually_ready_groups.append(group)
        if len(actually_ready_groups) >= max_n:
            break
    logger.info(f"{len(actually_ready_groups)} Level 2 XR files selected with necessary calibration data.")

    return actually_ready_groups


def level3_CIM_construct_flow_info(level2_files: list[File], level3_file: File, pipeline_config: dict,
                                   session=None, reference_time=None):
    flow_type = "level3_CIM"
    state = "planned"
    creation_time = datetime.now()
    priority = pipeline_config["flows"][flow_type]["priority"]["initial"]
    before_models = []
    after_models = []
    for file in level2_files:
        before_models.append(file.fcor_models[0].filename())
        after_models.append(file.fcor_models[1].filename())
    call_data = json.dumps(
        {
            "data_list": [level2_file.filename() for level2_file in level2_files],
            "before_f_corona_model_paths": before_models,
            "after_f_corona_model_paths": after_models,
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
    dates = [f.date_obs for f in level2_files]

    return [File(
                level="3",
                file_type="CI",
                observatory="M",
                file_version=pipeline_config["file_version"],
                software_version=__version__,
                date_obs=average_datetime(dates),
                state="planned",
                date_beg=min(dates),
                date_end=max(dates),
                outlier=any(f.outlier for f in level2_files),
                bad_packets=any(f.bad_packets for f in level2_files),
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
    for key in ["data_list", "before_f_corona_model_paths", "after_f_corona_model_paths"]:
        call_data[key] = file_name_to_full_path(call_data[key], pipeline_config["root"])
    return call_data


@flow
def level3_CIM_process_flow(flow_id: int | list[int], pipeline_config_path=None, session=None):
    # NOTE: this is not a typo... we're using the PIM core flow for this because it's flexible
    generic_process_flow_logic(flow_id, level3_PIM_CIM_flow, pipeline_config_path, session=session,
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
    input_file = input_files[0]

    return [File(
                level="3",
                file_type="CT",
                observatory="M",
                file_version=pipeline_config["file_version"],
                software_version=__version__,
                date_obs=input_file.date_obs,
                state="planned",
                date_beg=input_file.date_beg,
                date_end=input_file.date_end,
                outlier=input_file.outlier,
                bad_packets=input_file.bad_packets,
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

# TODO - repeat below logic for the final PAM set
@task(cache_policy=NO_CACHE)
def level3_CAM_query_ready_files(session, pipeline_config: dict, reference_time=None, max_n=100):
    return _level3_CAMPAM_query_ready_files(session, polarized=False, pipeline_config=pipeline_config, max_n=max_n)


@task(cache_policy=NO_CACHE)
def level3_PAM_query_ready_files(session, pipeline_config: dict, reference_time=None, max_n=100):
    return _level3_CAMPAM_query_ready_files(session, polarized=True, pipeline_config=pipeline_config, max_n=max_n)

def _level3_CAMPAM_query_ready_files(session, polarized: bool, pipeline_config: dict, reference_time=None, max_n=100):
    logger = get_run_logger()
    all_ready_files = (session.query(File)
                       .filter(File.state == "created")
                       .filter(File.level == "3")
                       .filter(File.file_type == ("PT" if polarized else "CT"))
                       .filter(File.observatory == "M")
                       .filter(File.outlier == 0)
                       .order_by(File.date_obs.desc()).all())
    # TODO - need to grab data from sets of rotation. look at movie processor for inspiration
    logger.info(f"{len(all_ready_files)} Level 3 {'P' if polarized else 'C'}TM files need to be processed to low-noise.")

    if len(all_ready_files) == 0:
        return []

    t0 = parse_datetime_str(pipeline_config["flows"]["level3_PAM" if polarized else "level3_CAM"]["t0"])
    increment = timedelta(minutes=32)

    end_time = t0
    # I'm sure there's a better way to do this, but let's step forward by increments to the present, and then we'll work
    # backwards back toward t0
    while end_time < datetime.now():
        end_time += increment
    start_time = end_time - increment

    grouped_files = []
    current_group = []
    while all_ready_files:
        file = all_ready_files.pop(0)
        if start_time <= file.date_obs < end_time:
            current_group.append(file)
        elif file.date_obs > end_time:
            # Shouldn't happen
            continue
        else:
            # file.date_obs < start_time, so this group is complete
            if current_group:
                ref_time = start_time + 0.5 * (end_time - start_time)
                ref_time = ref_time.replace(microsecond=0)
                # Check if we've already generated a (presumably incomplete) file for this date_obs.
                # TODO: it would be better to regenerate the file, but we don't have a way to do that sensibly now
                if not (session.query(File).filter(File.level == "3")
                        .filter(File.file_type == ("PT" if polarized else "CT"))
                        .filter(File.observatory == "M")
                        .filter(File.date_obs == ref_time)
                        .first()):
                    for f in current_group:
                        f._reference_time = ref_time
                    grouped_files.append(current_group)
            while not (start_time <= file.date_obs < end_time) and start_time >= t0:
                start_time -= increment
                end_time -= increment
            if start_time < t0:
                break
            current_group = [file]

    cutoff_time = (pipeline_config["flows"]["level3_PAM" if polarized else "level3_CAM"]
                   .get("ignore_missing_after_days", None))
    if cutoff_time is not None:
        cutoff_time = datetime.now(tz=UTC) - timedelta(days=cutoff_time)

    grouped_ready_files = []
    for group in grouped_files:
        group_is_complete = len(group) == (8 if polarized else 4)

        if len(grouped_ready_files) >= max_n:
            break

        if group_is_complete:
            grouped_ready_files.append(group)
            continue

        if cutoff_time and min(f.date_created for f in group).replace(tzinfo=UTC) < cutoff_time:
            # We've waited long enough. Just go ahead and make it.
            grouped_ready_files.append(group)
            continue

    logger.info(f"{len(grouped_ready_files)} groups heading out")
    return grouped_ready_files


def level3_CAMPAM_construct_flow_info(level3_files: list[File], level3_file_out: File,
                                   pipeline_config: dict, session=None, reference_time=None):
    flow_type = "level3_CAM" if level3_files[0].file_type[0] == "C" else "level3_PAM"
    state = "planned"
    creation_time = datetime.now(UTC)
    priority = pipeline_config["flows"][flow_type]["priority"]["initial"]
    reference_time = level3_files[0]._reference_time

    call_data = json.dumps(
        {
            "data_list": [
                os.path.join(level3_file.directory(pipeline_config["root"]), level3_file.filename())
                for level3_file in level3_files
            ],
            "reference_time": reference_time.strftime("%Y-%m-%dT%H:%M:%S"),
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


def level3_CAMPAM_construct_file_info(level3_files: list[File], pipeline_config: dict,
                                      reference_time=None) -> list[File]:
    reference_time = level3_files[0]._reference_time
    return [File(
                level="3",
                file_type="CA" if level3_files[0].file_type[0] == "C" else "PA",
                observatory="M",
                polarization="C" if level3_files[0].file_type[0] == "C" else "Y",
                file_version=pipeline_config["file_version"],
                software_version=__version__,
                date_obs=reference_time,
                date_beg=min([f.date_obs for f in level3_files if f.outlier == 0]),
                date_end=max([f.date_obs for f in level3_files if f.outlier == 0]),
                state="planned",
                # Outlier images are excluded from CAMs and PAMs
                outlier=0,
                bad_packets=False,
            )]


@flow
def level3_CAM_scheduler_flow(pipeline_config_path=None, session=None):
    generic_scheduler_flow_logic(
        level3_CAM_query_ready_files,
        level3_CAMPAM_construct_file_info,
        level3_CAMPAM_construct_flow_info,
        pipeline_config_path,
        session=session,
    )


@flow
def level3_CAM_process_flow(flow_id: int | list[int], pipeline_config_path=None, session=None):
    generic_process_flow_logic(flow_id, generate_level3_low_noise_flow, pipeline_config_path, session=session)

@flow
def level3_PAM_scheduler_flow(pipeline_config_path=None, session=None):
    generic_scheduler_flow_logic(
        level3_PAM_query_ready_files,
        level3_CAMPAM_construct_file_info,
        level3_CAMPAM_construct_flow_info,
        pipeline_config_path,
        session=session,
    )


@flow
def level3_PAM_process_flow(flow_id: int | list[int], pipeline_config_path=None, session=None):
    generic_process_flow_logic(flow_id, generate_level3_low_noise_flow, pipeline_config_path, session=session)
