import json
from datetime import datetime, timedelta

from dateutil.parser import parse as parse_datetime_str
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE

from punchbowl import __version__
from punchbowl.auto.control.db import File, Flow
from punchbowl.auto.control.processor import generic_process_flow_logic
from punchbowl.auto.control.scheduler import generic_scheduler_flow_logic
from punchbowl.auto.control.util import get_database_session, load_pipeline_configuration
from punchbowl.auto.flows.util import file_name_to_full_path
from punchbowl.level3.f_corona_model import construct_f_corona_model


def f_corona_background_query_ready_files(session, pipeline_config: dict, reference_time: datetime,
                                          reference_file: File):
    logger = get_run_logger()

    min_files_per_half = pipeline_config["flows"]["construct_f_corona_background"]["min_files_per_half"]
    max_files_per_half = pipeline_config["flows"]["construct_f_corona_background"]["max_files_per_half"]
    max_hours_per_half = pipeline_config["flows"]["construct_f_corona_background"]["max_hours_per_half"]
    t_start = reference_time - timedelta(hours=max_hours_per_half)
    t_end = reference_time + timedelta(hours=max_hours_per_half)

    target_file_type = reference_file.file_type[0] + "T"

    base_query = (session.query(File)
                  .filter(File.state.in_(["created", "progressed"]))
                  .filter(File.observatory == reference_file.observatory)
                  .filter(File.outlier == 0)
                  )

    first_half_inputs = (base_query
                         .filter(File.date_obs >= t_start)
                         .filter(File.date_obs <= reference_time)
                         .filter(File.file_type == target_file_type)
                         .filter(File.level == "2")
                         .order_by(File.date_obs.desc())
                         .limit(max_files_per_half).all())
    second_half_inputs = (base_query
                          .filter(File.date_obs >= reference_time)
                          .filter(File.date_obs <= t_end)
                          .filter(File.file_type == target_file_type)
                          .filter(File.level == "2")
                          .order_by(File.date_obs.asc())
                          .limit(max_files_per_half).all())

    enough_L2s = len(first_half_inputs) > min_files_per_half and len(second_half_inputs) > min_files_per_half
    if enough_L2s:
        all_ready_files = first_half_inputs + second_half_inputs

        logger.info(f"{len(all_ready_files)} Level 2 {target_file_type}{reference_file.observatory} files will be used "
                     "for F corona estimation.")
        return [f for f in all_ready_files]
    else:
        status = []
        status.append("not enough inputs")
        status.append(f"first half: {len(first_half_inputs)} files")
        status.append(f"second half: {len(second_half_inputs)} files")
        status.append(f"looked for inputs between {t_start.isoformat(' ')} and {t_end.isoformat(' ')}")
        logger.info(f'{reference_file.filename()}: ' + '; '.join(status))
    return []


@task(cache_policy=NO_CACHE)
def construct_f_corona_background_flow_info(level3_files: list[File],
                                            level3_f_model_file: [File],
                                            pipeline_config: dict,
                                            reference_time: datetime,
                                            file_type: str,
                                            spacecraft: str,
                                            session=None,
                                            ):
    flow_type = "construct_f_corona_background"
    state = "planned"
    creation_time = datetime.now()
    priority = pipeline_config["flows"][flow_type]["priority"]["initial"]
    call_data = json.dumps(
        {
            "filenames": [level3_file.filename() for level3_file in level3_files],
            "reference_time": str(reference_time),
            "polarized": level3_f_model_file[0].file_type[0] == "P",
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


@task(cache_policy=NO_CACHE)
def construct_f_corona_background_file_info(level2_files: list[File], pipeline_config: dict,
                                            reference_time: datetime, file_type: str,
                                    spacecraft: str) -> list[File]:
    date_obses = [f.date_obs for f in level2_files]

    return [File(
                level="3",
                file_type=file_type,
                observatory="M",
                polarization=level2_files[0].polarization,
                file_version=pipeline_config["file_version"],
                software_version=__version__,
                date_obs= reference_time,
                date_beg=min(date_obses),
                date_end=max(date_obses),
                state="planned",
            )]


@flow
def construct_f_corona_background_scheduler_flow(pipeline_config_path=None, session=None, reference_time: datetime | None = None):
    session = get_database_session()
    pipeline_config = load_pipeline_configuration(pipeline_config_path)
    logger = get_run_logger()

    if not pipeline_config["flows"]["construct_f_corona_background"].get("enabled", True):
        logger.info("Flow 'construct_f_corona_background' is not enabled---halting scheduler")
        return 0

    max_flows = pipeline_config["flows"]["construct_f_corona_background"].get("concurrency_limit", 1000)
    existing_flows = (session.query(Flow)
                      .where(Flow.flow_type == "construct_f_corona_background")
                      .where(Flow.state.in_(["planned", "launched", "running"])).count())
    flows_to_schedule = max_flows - existing_flows
    if flows_to_schedule <= 0:
        logger.info("Our maximum flow count has been reached; halting")
        return None
    logger.info(f"Will schedule up to {flows_to_schedule} flows")

    existing_models = (session.query(File)
                       .filter(File.level == "3")
                       .filter(File.file_type.in_(["CF", "PF"]))
                       .all())
    logger.info(f"There are {len(existing_models)} model records in the DB")

    existing_models = {(model.file_type, model.observatory, model.date_obs): model for model in existing_models}
    t0 = datetime.strptime(pipeline_config["flows"]["construct_f_corona_background"]["t0"], "%Y-%m-%d %H:%M:%S")
    increment = timedelta(hours=float(pipeline_config["flows"]["construct_f_corona_background"]["model_spacing_hours"]))
    n = 0
    models_to_try_creating = []
    # I'm sure there's a better way to do this, but let's step forward by increments to the present, and then we'll work
    # backwards back to t0, so that we prioritize the stray light models that QuickPUNCH uses
    while t0 + n * increment < datetime.now():
        n += 1

    for i in range(n, -1, -1):
        t = t0 + i * increment
        for model_type in ["CF", "PF"]:
            observatory = "M"
            key = (model_type, observatory, t)
            model = existing_models.get(key)
            if model is None:
                new_model = File(state="waiting",
                                 level="3",
                                 file_type=model_type,
                                 observatory=observatory,
                                 polarization=model_type[0],
                                 date_obs=t,
                                 date_created=datetime.now(),
                                 file_version=pipeline_config["file_version"],
                                 software_version=__version__)
                session.add(new_model)
                models_to_try_creating.append(new_model)
            elif model.state == "waiting":
                models_to_try_creating.append(model)

    session.commit()
    logger.info(f"There are {len(models_to_try_creating)} waiting models")


    target_date = pipeline_config.get("target_date", None)
    target_date = parse_datetime_str(target_date) if target_date else None
    if target_date:
        sorted_models = sorted(models_to_try_creating,
                                key=lambda x: abs((target_date - x.date_obs).total_seconds()))
    else:
        sorted_models = sorted(models_to_try_creating,
                                key=lambda x: x.date_obs,
                                reverse=True)

    to_schedule = []
    for model in sorted_models:
        ready_files = f_corona_background_query_ready_files(
            session, pipeline_config, model.date_obs, model)
        if ready_files:
            to_schedule.append((model, ready_files))
            logger.info(f"Will schedule {model.file_type} at {model.date_obs}")
            if len(to_schedule) == flows_to_schedule:
                break

    if to_schedule:
        for model, input_files in to_schedule:
            # Clear the placeholder model entry---it'll be regenerated in the scheduling flow
            args_dictionary = {"file_type": model.file_type, "spacecraft": model.observatory}
            dateobs = model.date_obs
            session.delete(model)
            generic_scheduler_flow_logic(
                lambda *args, **kwargs: [input_files],
                construct_f_corona_background_file_info,
                construct_f_corona_background_flow_info,
                pipeline_config,
                update_input_file_state=False,
                session=session,
                args_dictionary=args_dictionary,
                cap_planned_flows=False,
                reference_time=dateobs,
            )

        logger.info(f"Scheduled {len(to_schedule)} models")
    session.commit()


def construct_f_corona_call_data_processor(call_data: dict, pipeline_config, session=None) -> dict:
    call_data["filenames"] = file_name_to_full_path(call_data["filenames"], pipeline_config["root"])
    call_data["num_workers"] = 10
    call_data["num_loaders"] = 5
    return call_data

@flow
def construct_f_corona_background_process_flow(flow_id: int, pipeline_config_path=None, session=None):
    generic_process_flow_logic(flow_id, construct_f_corona_model, pipeline_config_path, session=session,
                               call_data_processor=construct_f_corona_call_data_processor)
