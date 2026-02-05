import json
from datetime import datetime, timedelta
from collections import defaultdict

from dateutil.parser import parse as parse_datetime_str
from prefect import flow, get_run_logger
from sqlalchemy import func

from punchbowl import __version__
from punchbowl.auto.control.db import File, Flow
from punchbowl.auto.control.processor import generic_process_flow_logic
from punchbowl.auto.control.scheduler import generic_scheduler_flow_logic
from punchbowl.auto.control.util import get_database_session, load_pipeline_configuration
from punchbowl.auto.flows.level2 import group_l2_inputs_single_observatory
from punchbowl.auto.flows.util import file_name_to_full_path
from punchbowl.level1.stray_light import estimate_polarized_stray_light, estimate_stray_light


def construct_clear_stray_light_check_for_inputs(session,
                                           pipeline_config: dict,
                                           reference_time: datetime,
                                           reference_file: File):
    logger = get_run_logger()

    min_files_per_half = pipeline_config["flows"]["construct_stray_light"]["min_files_per_half"]
    max_files_per_half = pipeline_config["flows"]["construct_stray_light"]["max_files_per_half"]
    max_hours_per_half = pipeline_config["flows"]["construct_stray_light"]["max_hours_per_half"]
    t_start = reference_time - timedelta(hours=max_hours_per_half)
    t_end = reference_time + timedelta(hours=max_hours_per_half)
    L0_impossible_after_days = pipeline_config["flows"]["construct_stray_light"]["new_L0_impossible_after_days"]
    more_L0_impossible = datetime.now() - t_end > timedelta(days=L0_impossible_after_days)

    file_type_mapping = {"SR": "XR", "SM": "YM", "SZ": "YZ", "SP": "YP"}
    target_file_type = file_type_mapping[reference_file.file_type]
    L0_type_mapping = {"SR": "CR", "SM": "PM", "SZ": "PZ", "SP": "PP"}
    L0_target_file_type = L0_type_mapping[reference_file.file_type]

    base_query = (session.query(File)
                  .filter(File.state.in_(["created", "progressed"]))
                  .filter(File.observatory == reference_file.observatory)
                  .filter(~File.outlier)
                  )

    first_half_inputs = (base_query
                         .filter(File.date_obs >= t_start)
                         .filter(File.date_obs <= reference_time)
                         .filter(File.file_type == target_file_type)
                         .filter(File.level == "1")
                         .order_by(File.date_obs.desc())
                         .limit(max_files_per_half).all())

    second_half_inputs = (base_query
                          .filter(File.date_obs >= reference_time)
                          .filter(File.date_obs <= t_end)
                          .filter(File.file_type == target_file_type)
                          .filter(File.level == "1")
                          .order_by(File.date_obs.asc())
                          .limit(max_files_per_half).all())

    first_half_L0s = (base_query
                      .filter(File.date_obs >= t_start)
                      .filter(File.date_obs <= reference_time)
                      .filter(File.file_type == L0_target_file_type)
                      .filter(File.level == "0")
                      .order_by(File.date_obs.desc())
                      .limit(max_files_per_half).all())

    second_half_L0s = (base_query
                       .filter(File.date_obs >= reference_time)
                       .filter(File.date_obs <= t_end)
                       .filter(File.file_type == L0_target_file_type)
                       .filter(File.level == "0")
                       .order_by(File.date_obs.asc())
                       .limit(max_files_per_half).all())

    # Allow 5% of the L0s to not be processed, in case a few fail
    all_inputs_ready = (len(first_half_inputs) >= 0.95 * len(first_half_L0s)
                        and len(second_half_inputs) >= 0.95 * len(second_half_L0s))
    enough_L1s = len(first_half_inputs) > min_files_per_half and len(second_half_inputs) > min_files_per_half
    max_L1s = len(first_half_inputs) == max_files_per_half and len(second_half_inputs) == max_files_per_half

    produce = False
    if more_L0_impossible:
        if len(first_half_L0s) < min_files_per_half or len(second_half_L0s) < min_files_per_half:
            reference_file.state = "impossible"
            # Record who deemed this to be impossible
            reference_file.file_version = pipeline_config["file_version"]
            reference_file.software_version = __version__
            reference_file.date_created = datetime.now()
            logger.info(f"{reference_file.filename()} marked impossible")
        elif all_inputs_ready and enough_L1s:
            n = min(len(first_half_inputs), len(second_half_inputs))
            first_half_inputs = first_half_inputs[:n]
            second_half_inputs = second_half_inputs[:n]
            produce = True
    elif max_L1s:
        produce = True

    if produce:
        all_ready_files = first_half_inputs + second_half_inputs

        logger.info(f"{len(all_ready_files)} Level 1 {target_file_type}{reference_file.observatory} files will be used "
                     "for stray light estimation.")
        return [f.file_id for f in all_ready_files]
    else:
        status = []
        if not all_inputs_ready:
            status.append("more L0s than L1s---waiting for L1s to be produced")
        if not enough_L1s:
            status.append("not enough inputs")
        status.append(f"{'not' if more_L0_impossible else ''} waiting for more downlinks")
        status.append(f"first half: {len(first_half_inputs)} files, {len(first_half_L0s)} L0s")
        status.append(f"second half: {len(second_half_inputs)} files, {len(second_half_L0s)} L0s")
        status.append(f"looked for inputs between {t_start.isoformat(' ')} and {t_end.isoformat(' ')}")
        logger.info(f'{reference_file.filename()}: ' + '; '.join(status))
    return []


def construct_polarized_stray_light_check_for_inputs(session,
                                           pipeline_config: dict,
                                           reference_time: datetime,
                                           reference_files: list[File]):
    logger = get_run_logger()

    min_files_per_half = pipeline_config["flows"]["construct_stray_light"]["min_files_per_half"]
    max_files_per_half = pipeline_config["flows"]["construct_stray_light"]["max_files_per_half"]
    max_hours_per_half = pipeline_config["flows"]["construct_stray_light"]["max_hours_per_half"]
    t_start = reference_time - timedelta(hours=max_hours_per_half)
    t_end = reference_time + timedelta(hours=max_hours_per_half)
    L0_impossible_after_days = pipeline_config["flows"]["construct_stray_light"]["new_L0_impossible_after_days"]
    more_L0_impossible = datetime.now() - t_end > timedelta(days=L0_impossible_after_days)

    target_file_types = ("YP", "YM", "YZ")
    L0_target_file_types = ("PP", "PM", "PZ")

    base_query = (session.query(File)
                  .filter(File.state.in_(["created", "progressed"]))
                  .filter(File.observatory == reference_files[0].observatory)
                  .filter(~File.outlier)
                  )

    first_half_inputs = (base_query
                         .filter(File.date_obs >= t_start)
                         .filter(File.date_obs <= reference_time)
                         .filter(File.file_type.in_(target_file_types))
                         .filter(File.level == "1")
                         .order_by(File.date_obs.asc()).all())

    second_half_inputs = (base_query
                          .filter(File.date_obs >= reference_time)
                          .filter(File.date_obs <= t_end)
                          .filter(File.file_type.in_(target_file_types))
                          .filter(File.level == "1")
                          .order_by(File.date_obs.asc()).all())

    first_half_L0s = (base_query
                      .filter(File.date_obs >= t_start)
                      .filter(File.date_obs <= reference_time)
                      .filter(File.file_type.in_(L0_target_file_types))
                      .filter(File.level == "0")
                      .order_by(File.date_obs.asc()).all())

    second_half_L0s = (base_query
                       .filter(File.date_obs >= reference_time)
                       .filter(File.date_obs <= t_end)
                       .filter(File.file_type.in_(L0_target_file_types))
                       .filter(File.level == "0")
                       .order_by(File.date_obs.asc()).all())

    order = "MZP" if reference_files[0].observatory == "4" else "PZM"
    first_half_inputs = group_l2_inputs_single_observatory(first_half_inputs, order, only_complete=True)[::-1]
    second_half_inputs = group_l2_inputs_single_observatory(second_half_inputs, order, only_complete=True)
    first_half_L0s = group_l2_inputs_single_observatory(first_half_L0s, order, only_complete=True)[::-1]
    second_half_L0s = group_l2_inputs_single_observatory(second_half_L0s, order, only_complete=True)

    # Allow 5% of the L0s to not be processed, in case a few fail
    all_inputs_ready = (len(first_half_inputs) >= 0.95 * len(first_half_L0s)
                        and len(second_half_inputs) >= 0.95 * len(second_half_L0s))
    enough_L1s = len(first_half_inputs) > min_files_per_half and len(second_half_inputs) > min_files_per_half
    max_L1s = len(first_half_inputs) >= max_files_per_half and len(second_half_inputs) >= max_files_per_half

    produce = False
    if more_L0_impossible:
        if len(first_half_L0s) < min_files_per_half or len(second_half_L0s) < min_files_per_half:
            for reference_file in reference_files:
                reference_file.state = "impossible"
                # Record who deemed this to be impossible
                reference_file.file_version = pipeline_config["file_version"]
                reference_file.software_version = __version__
                reference_file.date_created = datetime.now()
            logger.info(f"{[f.filename() for f in reference_files]} marked impossible")
        elif all_inputs_ready and enough_L1s:
            n = min(len(first_half_inputs), len(second_half_inputs))
            first_half_inputs = first_half_inputs[:n]
            second_half_inputs = second_half_inputs[:n]
            produce = True
    elif max_L1s:
        produce = True

    if produce:
        all_ready_files = []
        for group in first_half_inputs[:max_files_per_half]:
            all_ready_files.extend(group)
        for group in second_half_inputs[:max_files_per_half]:
            all_ready_files.extend(group)

        logger.info(f"{len(all_ready_files)} Level 1 Y*{reference_files[0].observatory} files will be used "
                     "for stray light estimation.")
        return [f.file_id for f in all_ready_files]
    else:
        status = []
        if not all_inputs_ready:
            status.append("more L0s than L1s---waiting for L1s to be produced")
        if not enough_L1s:
            status.append("not enough inputs")
        status.append(f"{'not' if more_L0_impossible else ''} waiting for more downlinks")
        status.append(f"first half: {len(first_half_inputs)} files, {len(first_half_L0s)} L0s")
        status.append(f"second half: {len(second_half_inputs)} files, {len(second_half_L0s)} L0s")
        status.append(f"looked for inputs between {t_start.isoformat(' ')} and {t_end.isoformat(' ')}")
        logger.info(f'{[f.filename() for f in reference_files]}: ' + '; '.join(status))
    return []


def construct_stray_light_flow_info(level1_files: list[File],
                                    level1_stray_light_files: File,
                                    pipeline_config: dict,
                                    reference_time: datetime,
                                    file_type: str,
                                    spacecraft: str,
                                    session=None):
    flow_type = "construct_stray_light"
    state = "planned"
    creation_time = datetime.now()
    priority = pipeline_config["flows"][flow_type]["priority"]["initial"]
    call_data = json.dumps(
        {
            "filepaths": [level1_file.filename() for level1_file in level1_files],
            "reference_time": reference_time.strftime("%Y-%m-%d %H:%M:%S"),
            "spacecraft": spacecraft,
            "is_polarized": file_type != ["SR"],
        },
    )
    return Flow(
        flow_type=flow_type,
        state=state,
        flow_level="1",
        creation_time=creation_time,
        priority=priority,
        call_data=call_data,
    )


def construct_stray_light_file_info(level1_files: list[File],
                                    pipeline_config: dict,
                                    reference_time: datetime,
                                    file_type: str,
                                    spacecraft: str) -> list[File]:
    date_obses = [f.date_obs for f in level1_files]
    date_beg, date_end = min(date_obses), max(date_obses)
    if file_type == ["SR"]:
        return [File(
                    level="1",
                    file_type="SR",
                    observatory=spacecraft,
                    polarization="C",
                    file_version=pipeline_config["file_version"],
                    software_version=__version__,
                    date_obs=reference_time,
                    date_beg=date_beg,
                    date_end=date_end,
                    state="planned",
                )]
    return [File(
                level="1",
                file_type="SM",
                observatory=spacecraft,
                polarization="M",
                file_version=pipeline_config["file_version"],
                software_version=__version__,
                date_obs=reference_time,
                date_beg=date_beg,
                date_end=date_end,
                state="planned",
            ),
            File(
                level="1",
                file_type="SZ",
                observatory=spacecraft,
                polarization="Z",
                file_version=pipeline_config["file_version"],
                software_version=__version__,
                date_obs=reference_time,
                date_beg=date_beg,
                date_end=date_end,
                state="planned",
            ),
            File(
                level="1",
                file_type="SP",
                observatory=spacecraft,
                polarization="P",
                file_version=pipeline_config["file_version"],
                software_version=__version__,
                date_obs=reference_time,
                date_beg=date_beg,
                date_end=date_end,
                state="planned",
            )]

@flow
def construct_stray_light_scheduler_flow(pipeline_config_path=None, session=None, reference_time: datetime | None = None):
    session = get_database_session()
    pipeline_config = load_pipeline_configuration(pipeline_config_path)
    logger = get_run_logger()

    if not pipeline_config["flows"]["construct_stray_light"].get("enabled", True):
        logger.info("Flow 'construct_stray_light' is not enabled---halting scheduler")
        return

    max_flows = pipeline_config["flows"]["construct_stray_light"].get("concurrency_limit", 1000)
    existing_flows = (session.query(Flow)
                      .where(Flow.flow_type == "construct_stray_light")
                      .where(Flow.state.in_(["planned", "launched", "running"])).count())

    flows_to_schedule = max_flows - existing_flows
    if flows_to_schedule <= 0:
        logger.info("Our maximum flow count has been reached; halting")
        return
    logger.info(f"Will schedule up to {flows_to_schedule} flows")

    existing_models = (session.query(File)
                       .filter(File.level == "1")
                       .filter(File.file_type.in_(["SR", "SM", "SZ", "SP"]))
                       .all())
    logger.info(f"There are {len(existing_models)} model records in the DB")

    existing_models = {(model.file_type, model.observatory, model.date_obs): model for model in existing_models}
    t0 = datetime.strptime(pipeline_config["flows"]["construct_stray_light"]["t0"], "%Y-%m-%d %H:%M:%S")
    increment = timedelta(hours=float(pipeline_config["flows"]["construct_stray_light"]["model_spacing_hours"]))

    n = 0
    # I'm sure there's a better way to do this, but let's step forward by increments to the present, and then we'll work
    # backwards back to t0, so that we prioritize the stray light models that QuickPUNCH uses
    while t0 + n * increment < datetime.now():
        n += 1

    for i in range(n, -1, -1):
        t = t0 + i * increment
        for model_type in ["SR", "SM", "SZ", "SP"]:
            for observatory in ["1", "2", "3", "4"]:
                key = (model_type, observatory, t)
                model = existing_models.get(key)
                if model is None:
                    new_model = File(state="waiting",
                                     level="1",
                                     file_type=model_type,
                                     observatory=observatory,
                                     polarization="C" if model_type[1] == "R" else model_type[1],
                                     date_obs=t,
                                     date_created=datetime.now(),
                                     file_version=pipeline_config["file_version"],
                                     software_version=__version__)
                    session.add(new_model)
                    existing_models[key] = new_model

    waiting_models_by_time_and_type = defaultdict(list)
    for model in existing_models.values():
        if model.state == "waiting":
            is_polarized = model.polarization != "C"
            waiting_models_by_time_and_type[(model.date_obs, model.observatory, is_polarized)].append(model)

    logger.info(f"There are {len(waiting_models_by_time_and_type)} waiting models")

    dates = (session.query(func.min(File.date_obs), func.max(File.date_obs))
             .where(File.file_type.in_(["XR", "YZ", "YP", "YM"]))
             .where(File.state.in_(["progressed", "created"])).all())

    if dates[0][0] is None:
        logger.info("There are no X files in the database")
        session.commit()
        return

    earliest_input, latest_input = dates[0]

    target_date = pipeline_config.get("target_date", None)
    target_date = parse_datetime_str(target_date) if target_date else None
    if target_date:
        sorted_models = sorted(waiting_models_by_time_and_type.items(),
                                key=lambda x: abs((target_date - x[0][0]).total_seconds()))
    else:
        sorted_models = sorted(waiting_models_by_time_and_type.items(),
                                key=lambda x: x[0][0],
                                reverse=True)
    n_skipped = 0
    to_schedule = []
    for (date_obs, observatory, is_polarized), models in sorted_models:
        if not (earliest_input <= date_obs <= latest_input):
            n_skipped += 1
            continue
        if is_polarized:
            if len(models) != 3:
                logger.warning(f"Wrong number of waiting polarized models for {date_obs}, got {len(models)}---skipping")
                continue
            ready_files = construct_polarized_stray_light_check_for_inputs(
                session, pipeline_config, models[0].date_obs, models)
            if ready_files:
                to_schedule.append((models, ready_files))
                logger.info(f"Will schedule {' '.join(m.file_type + m.observatory for m in models)} "
                            f"at {models[0].date_obs}")
                if len(to_schedule) == flows_to_schedule:
                    break
        else:
            if len(models) != 1:
                logger.warning(f"Wrong number of waiting clear models for {date_obs}, got {len(models)}---skipping")
                continue
            model = models[0]
            ready_files = construct_clear_stray_light_check_for_inputs(
                session, pipeline_config, model.date_obs, model)
            if ready_files:
                to_schedule.append(([model], ready_files))
                logger.info(f"Will schedule {model.file_type}{model.observatory} at {model.date_obs}")
                if len(to_schedule) == flows_to_schedule:
                    break

    logger.info(f"{n_skipped} models fall outside the range of existing X files and were not queried")

    if to_schedule:
        for models, input_files in to_schedule:
            # Clear the placeholder model entry---it'll be regenerated in the scheduling flow
            args_dictionary = {"file_type": [m.file_type for m in models], "spacecraft": models[0].observatory}
            dateobs = models[0].date_obs
            for model in models:
                session.delete(model)
            generic_scheduler_flow_logic(
                lambda *args, **kwargs: [input_files],
                construct_stray_light_file_info,
                construct_stray_light_flow_info,
                pipeline_config,
                update_input_file_state=False,
                session=session,
                args_dictionary=args_dictionary,
                cap_planned_flows=False,
                reference_time=dateobs,
            )

        logger.info(f"Scheduled {len(to_schedule)} models")
    session.commit()


def construct_stray_light_call_data_processor(call_data: dict, pipeline_config, session) -> dict:
    # Prepend the directory path to each input file
    call_data["filepaths"] = file_name_to_full_path(call_data["filepaths"], pipeline_config["root"])
    if call_data["is_polarized"]:
        call_data["zfilepaths"] = call_data["filepaths"][1::3]
        if call_data["spacecraft"] == "4":
            call_data["mfilepaths"] = call_data["filepaths"][::3]
            call_data["pfilepaths"] = call_data["filepaths"][2::3]
        else:
            call_data["pfilepaths"] = call_data["filepaths"][::3]
            call_data["mfilepaths"] = call_data["filepaths"][2::3]
        del call_data["filepaths"]
    del call_data["spacecraft"]
    call_data["num_workers"] = 40
    call_data["num_loaders"] = 12
    return call_data


@flow
def construct_stray_light_process_flow(flow_id: int, pipeline_config_path=None, session=None):
    generic_process_flow_logic(flow_id, call_correct_stray_light_function, pipeline_config_path, session=session,
                               call_data_processor=construct_stray_light_call_data_processor)


def call_correct_stray_light_function(*args, **kwargs):
    is_polarized = kwargs.pop("is_polarized")
    if is_polarized:
        return estimate_polarized_stray_light(*args, **kwargs)
    return estimate_stray_light(*args, **kwargs)
