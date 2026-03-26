import json
import random
from datetime import datetime, timedelta
from collections import defaultdict

from dateutil.parser import parse as parse_datetime_str
from prefect import flow, get_run_logger
from sqlalchemy import func, text

from punchbowl import __version__
from punchbowl.auto.control.db import File, Flow
from punchbowl.auto.control.processor import generic_process_flow_logic
from punchbowl.auto.control.scheduler import generic_scheduler_flow_logic
from punchbowl.auto.control.util import batched, get_database_session, load_pipeline_configuration
from punchbowl.auto.flows.level1 import get_mask_file
from punchbowl.auto.flows.level2 import group_l2_inputs
from punchbowl.auto.flows.util import file_name_to_full_path
from punchbowl.level1.stray_light import estimate_stray_light


def construct_stray_light_check_for_inputs(session,
                                           pipeline_config: dict,
                                           reference_time: datetime,
                                           reference_files: list[File]):
    logger = get_run_logger()

    polarized = reference_files[0].file_type != "SR"
    pol_type = 'pol' if polarized else 'clear'
    min_files_per_half = pipeline_config["flows"]["construct_stray_light"][f"{pol_type}_min_files_per_half"]
    max_files_per_half = pipeline_config["flows"]["construct_stray_light"][f"{pol_type}_max_files_per_half"]
    max_hours_per_half = pipeline_config["flows"]["construct_stray_light"][f"{pol_type}_max_hours_per_half"]
    file_stride = pipeline_config["flows"]["construct_stray_light"][f"{pol_type}_file_stride"]
    t_start = reference_time - timedelta(hours=max_hours_per_half)
    t_end = reference_time + timedelta(hours=max_hours_per_half)
    L0_impossible_after_days = pipeline_config["new_L0_impossible_after_days"]
    more_L0_impossible = datetime.now() - t_end > timedelta(days=L0_impossible_after_days)

    if reference_files[0].observatory == '4':
        file_type_mapping = {"SR": "XR", "SM": "XM", "SZ": "XZ", "SP": "XP"}
    else:
        file_type_mapping = {"SR": "XR", "SM": "YM", "SZ": "YZ", "SP": "YP"}

    out_types = ("M", "Z", "P") if polarized else ("R")
    target_file_types = [file_type_mapping["S" + t] for t in out_types]
    L0_type_mapping = {"SR": "CR", "SM": "PM", "SZ": "PZ", "SP": "PP"}
    L0_target_file_types = [L0_type_mapping["S" + t] for t in out_types]

    count_multiplier = 3 if polarized else 1
    count_multiplier *= file_stride

    base_query = (session.query(File)
                  .filter(File.state.in_(["created", "progressed"]))
                  .filter(File.observatory == reference_files[0].observatory)
                  .filter(~File.bad_packets)
                  )

    first_half_inputs = (base_query
                         .filter(File.date_obs >= t_start)
                         .filter(File.date_obs <= reference_time)
                         .filter(File.file_type.in_(target_file_types))
                         .filter(File.level == "1")
                         .order_by(File.date_obs.desc())
                         .limit(max_files_per_half * count_multiplier).all())

    second_half_inputs = (base_query
                          .filter(File.date_obs >= reference_time)
                          .filter(File.date_obs <= t_end)
                          .filter(File.file_type.in_(target_file_types))
                          .filter(File.level == "1")
                          .order_by(File.date_obs.asc())
                          .limit(max_files_per_half * count_multiplier).all())

    first_half_L0s = (base_query
                      .filter(File.date_obs >= t_start)
                      .filter(File.date_obs <= reference_time)
                      .filter(File.file_type.in_(L0_target_file_types))
                      .filter(File.level == "0")
                      .order_by(File.date_obs.desc())
                      .limit(max_files_per_half * count_multiplier).all())

    second_half_L0s = (base_query
                       .filter(File.date_obs >= reference_time)
                       .filter(File.date_obs <= t_end)
                       .filter(File.file_type.in_(L0_target_file_types))
                       .filter(File.level == "0")
                       .order_by(File.date_obs.asc())
                       .limit(max_files_per_half * count_multiplier).all())

    # Allow 5% of the L0s to not be processed, in case a few fail
    all_inputs_ready = (len(first_half_inputs) >= 0.95 * len(first_half_L0s)
                        and len(second_half_inputs) >= 0.95 * len(second_half_L0s))

    if polarized:
        first_half_groups = group_l2_inputs(first_half_inputs[::-1])
        second_half_groups = group_l2_inputs(second_half_inputs)
    else:
        first_half_groups = [[f] for f in first_half_inputs]
        second_half_groups = [[f] for f in second_half_inputs]

    if file_stride > 1:
        random.seed(1)
        # Apply a stride that doesn't phase weirdly with where we are in a roll position
        first_half_groups = [random.choice(pair) for pair in batched(first_half_groups, file_stride)]
        second_half_groups = [random.choice(pair) for pair in batched(second_half_groups, file_stride)]

    enough_L1s = len(first_half_groups) > min_files_per_half and len(second_half_groups) > min_files_per_half
    max_L1s = len(first_half_groups) == max_files_per_half and len(second_half_groups) == max_files_per_half

    produce = False
    if more_L0_impossible:
        if (len(first_half_L0s) < min_files_per_half * count_multiplier
                or len(second_half_L0s) < min_files_per_half * count_multiplier):
            for reference_file in reference_files:
                reference_file.state = "impossible"
                # Record who deemed this to be impossible
                reference_file.file_version = pipeline_config["file_version"]
                reference_file.software_version = __version__
                reference_file.date_created = datetime.now()
                logger.info(f"{reference_file.filename()} marked impossible")
        elif all_inputs_ready and enough_L1s:
            # Since the two lists of groups are in nearest-to-the-reference-date first order, cutting them to the
            # same size helps ensure the range they cover is symmetric around the ref date.
            n = min(len(first_half_groups), len(second_half_groups))
            first_half_groups = first_half_groups[:n]
            second_half_groups = second_half_groups[:n]
            produce = True
    elif max_L1s:
        produce = True

    if produce:
        all_ready_files = first_half_groups + second_half_groups
        all_ready_files = [f for group in all_ready_files for f in group]

        logger.info(f"{len(all_ready_files)} Level 1 {','.join(target_file_types)}{reference_files[0].observatory} "
                    "files will be used for stray light estimation.")
        return [f.file_id for f in all_ready_files]
    else:
        status = []
        if not all_inputs_ready:
            status.append("more L0s than L1s---waiting for L1s to be produced")
        if not enough_L1s:
            status.append("not enough inputs")
        status.append(f"{'not' if more_L0_impossible else ''} waiting for more downlinks")
        status.append(f"first half: {len(first_half_inputs)} files, {len(first_half_groups)} groups, "
                      f"{len(first_half_L0s)} L0s")
        status.append(f"second half: {len(second_half_inputs)} files, {len(second_half_groups)} groups, "
                      f"{len(second_half_L0s)} L0s")
        status.append(f"looked for inputs between {t_start.isoformat(' ')} and {t_end.isoformat(' ')}")
        logger.info(f'{', '.join(rf.filename() for rf in reference_files)}: ' + '; '.join(status))
    return []


def construct_stray_light_flow_info(level1_files: list[File],
                                    level1_stray_light_files: File,
                                    pipeline_config: dict,
                                    reference_time: datetime,
                                    is_polarized: bool,
                                    spacecraft: str,
                                    session=None):
    flow_type = "construct_stray_light"
    state = "planned"
    creation_time = datetime.now()
    priority = pipeline_config["flows"][flow_type]["priority"]["initial"]
    mask = get_mask_file(level1_files[0], pipeline_config, session=session)
    pol_type = 'pol' if is_polarized else 'clear'

    dt = func.abs(func.timestampdiff(text("second"), File.date_obs, reference_time))
    # TODO: do we need this?
    # nearest_created_model = (
    #         session.query(File)
    #         .where(File.file_type == file_type)
    #         .where(File.observatory == spacecraft)
    #         .where(File.state == 'created')
    #         .order_by(dt.asc()).first())
    nearest_created_model = None

    call_data = json.dumps(
        {
            "filepaths": [level1_file.filename() for level1_file in level1_files],
            "reference_time": reference_time.strftime("%Y-%m-%d %H:%M:%S"),
            "spacecraft": spacecraft,
            "image_mask_path": mask.filename().replace(".fits", ".bin"),
            "window_size": pipeline_config["flows"][flow_type][f"{pol_type}_neighborhood_size"],
            "fallback_model_path": None if nearest_created_model is None else nearest_created_model.filename(),
            "polarized": is_polarized,
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
                                    is_polarized: bool,
                                    spacecraft: str) -> list[File]:
    date_obses = [f.date_obs for f in level1_files]
    date_beg, date_end = min(date_obses), max(date_obses)
    polarizations = ("M", "Z", "P") if is_polarized else ("C",)
    return [File(
                level="1",
                file_type="SR" if pol == "C" else "S" + pol,
                observatory=spacecraft,
                polarization=pol,
                file_version=pipeline_config["file_version"],
                software_version=__version__,
                date_obs=reference_time,
                date_beg=date_beg,
                date_end=date_end,
                state="planned",
            ) for pol in polarizations]

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

    waiting_models = [model for model in existing_models.values() if model.state == 'waiting']

    logger.info(f"There are {len(waiting_models)} waiting models")

    waiting_groups = defaultdict(list)
    for model in waiting_models:
        waiting_groups[(model.date_obs, model.polarization == "C", model.observatory)].append(model)
    waiting_groups = list(waiting_groups.values())

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
        sorted_models = sorted(waiting_groups,
                               key=lambda group: abs((target_date - group[0].date_obs).total_seconds()))
    else:
        sorted_models = sorted(waiting_groups,
                               key=lambda group: group[0].date_obs,
                               reverse=True)
    n_skipped = 0
    to_schedule = []
    for model_group in sorted_models:
        if not (earliest_input <= model_group[0].date_obs <= latest_input):
            n_skipped += 1
            continue
        ready_files = construct_stray_light_check_for_inputs(
            session, pipeline_config, model_group[0].date_obs, model_group)
        if ready_files:
            to_schedule.append((model_group, ready_files))
            codes = [f"{model.file_type}{model.observatory}" for model in model_group]
            logger.info(f"Will schedule {', '.join(codes)} at {model_group[0].date_obs}")
            if len(to_schedule) == flows_to_schedule:
                break

    logger.info(f"{n_skipped} models fall outside the range of existing X files and were not queried")

    if to_schedule:
        for model_group, input_files in to_schedule:
            args_dictionary = {"is_polarized": len(model_group) > 1, "spacecraft": model_group[0].observatory}
            dateobs = model_group[0].date_obs
            # Clear the placeholder model entry---it'll be regenerated in the scheduling flow
            for model in model_group:
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
    for key in ["filepaths", "image_mask_path", "fallback_model_path"]:
        call_data[key] = file_name_to_full_path(call_data[key], pipeline_config["root"])
    del call_data["spacecraft"]
    call_data["num_workers"] = 30
    return call_data


@flow
def construct_stray_light_process_flow(flow_id: int, pipeline_config_path=None, session=None):
    generic_process_flow_logic(flow_id, estimate_stray_light, pipeline_config_path, session=session,
                               call_data_processor=construct_stray_light_call_data_processor)
