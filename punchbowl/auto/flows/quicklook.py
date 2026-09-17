import os
import json
from datetime import UTC, datetime, timedelta

from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from prefect.context import get_run_context
from prefect.runtime import flow_run

from punchbowl.auto.control.db import File, Flow, Quicklook
from punchbowl.auto.control.util import get_database_session, load_pipeline_configuration, load_quicklook_scaling
from punchbowl.auto.flows.util import file_name_to_full_path
from punchbowl.data.punch_io import load_ndcube_from_fits, write_ndcube_to_quicklook
from punchbowl.data.visualize import animate_punch
from punchbowl.prefect import get_logger


@task(cache_policy=NO_CACHE)
def visualize_query_ready_files(session,
                                pipeline_config: dict,
                                reference_time: datetime) -> tuple[list, list]:
    logger = get_logger()

    all_ready_files = []
    all_product_codes = []
    all_tasks = []

    code_mapping = {"3": ["CA", "PA", "CT", "PT"],
                    "Q": ["QA", "QN"]}

    expected_files = pipeline_config["flows"]["quicklook"]["expected_files"]

    day = datetime.fromisoformat(pipeline_config["flows"]["quicklook"]["start_time"])
    reference_time = reference_time.replace(hour=0, minute=0, second=0, microsecond=0)

    while day <= reference_time:
        day = day + timedelta(days=1)
        for level, codes in code_mapping.items():
            for product_code in codes:
                quicklook_results = (session.query(Quicklook)
                                     .filter(Quicklook.day == day)
                                     .filter(Quicklook.level == level)
                                     .filter(Quicklook.code == product_code).all())
                image_made = quicklook_results.image_made if quicklook_results else False
                movie_made = quicklook_results.movie_made if quicklook_results else False

                if image_made and movie_made:
                    continue

                files = (session.query(File)
                        .filter(File.state.in_(["created", "progressed", "quickpunched"]))
                        .filter(File.date_obs >= day)
                        .filter(File.date_obs < day + timedelta(days=1))
                        .filter(File.level == level)
                        .filter(File.file_type == product_code[0:2])
                        .filter(File.observatory == product_code[2])
                        .order_by(File.date_obs.asc()).all())

                movie_nfiles = expected_files[f"L{level}_{product_code}"]

                make_image = (not image_made) and (len(files) > 0)
                make_movie = (not movie_made) and (len(files) > movie_nfiles)

                if make_image or make_movie:
                    all_ready_files.append(list(files))
                    all_product_codes.append(f"L{level}_{product_code}")
                    all_tasks.append({"day": day, "level": level, "code": product_code,
                                    "make_image": make_image, "make_movie": make_movie})


    logger.info(f"{len(all_product_codes)} days will be visualized.")
    return all_ready_files, all_product_codes, all_tasks


@task(cache_policy=NO_CACHE)
def visualize_flow_info(input_files: list[File],
                        product_code: str,
                        pipeline_config: dict,
                        task_item: dict,
                        framerate: int = 10,
                        resolution: int = 1024,
                        ):
    flow_type = "movie"
    state = "planned"

    creation_time = datetime.now()
    out_path = creation_time.strftime("%Y/%m/%d")

    priority = pipeline_config["flows"][flow_type]["priority"]["initial"]
    call_data = json.dumps(
        {
            "file_list": [input_file.filename() for input_file in input_files],
            "product_code": product_code,
            "day": task_item["day"].isoformat(),
            "level": task_item["level"],
            "code": task_item["code"],
            "make_image": task_item["make_image"],
            "make_movie": task_item["make_movie"],
            "output_movie_dir": os.path.join("movies", out_path),
            "framerate": framerate,
            "resolution": resolution,
            "ffmpeg_cmd": pipeline_config["flows"]["movie"]["options"].get("ffmpeg_cmd", "ffmpeg"),
        },
    )
    return Flow(
        flow_type=flow_type,
        state=state,
        flow_level="M",
        creation_time=creation_time,
        priority=priority,
        call_data=call_data,
    )


@flow
def quicklook_scheduler_flow(pipeline_config_path=None,
                             session=None,
                             reference_time: datetime | None = None,
                             framerate: int = 10,
                             resolution: int = 1024):
    if session is None:
        session = get_database_session()

    reference_time = reference_time or datetime.now(UTC)

    pipeline_config = load_pipeline_configuration(pipeline_config_path)

    file_lists, product_codes, tasks = visualize_query_ready_files(
        session, pipeline_config, reference_time,
    )

    for file_list, product_code, task in zip(file_lists, product_codes, tasks):
        flow = visualize_flow_info(file_list, product_code, pipeline_config, task,
                                   framerate=framerate, resolution=resolution)
        session.add(flow)

    session.commit()

def generate_flow_run_name():
    parameters = flow_run.parameters
    code = parameters["product_code"]
    files = parameters["file_list"]
    return f"movie-{code}-len={len(files)}-{datetime.now()}"


@flow(flow_run_name=generate_flow_run_name)
def quicklook_core_flow(file_list: list,
                        output_movie_dir: str,
                        make_image: bool,
                        make_movie: bool,
                        framerate: int = 10) -> None:
    cube = load_ndcube_from_fits(file_list[0])
    vmin, vmax = load_quicklook_scaling(level=cube.meta["LEVEL"].value, product=cube.meta["TYPECODE"].value, obscode=cube.meta["OBSCODE"].value)

    path_image = os.path.join(output_movie_dir, f"PUNCH_{cube.meta["TYPECODE"].value}{cube.meta["OBSCODE"].value}_{cube.meta.datetime.strftime("%Y%m%d")}_v{cube.meta["FILEVRSN"].value}.jpg")
    path_movie = os.path.join(output_movie_dir, f"PUNCH_{cube.meta["TYPECODE"].value}{cube.meta["OBSCODE"].value}_{cube.meta.datetime.strftime("%Y%m%d")}_v{cube.meta["FILEVRSN"].value}.mp4")

    os.makedirs(os.path.dirname(path_image), exist_ok=True)

    if make_image:
        write_ndcube_to_quicklook(cube, filename=path_image, vmin=vmin, vmax=vmax)
    if make_movie:
        animate_punch(file_list, output_path=path_movie, fps=framerate, n_jobs=12, vmin=vmin, vmax=vmax)


@flow
def quicklook_process_flow(flow_id: int, pipeline_config_path=None, session=None):
    if session is None:
        session = get_database_session()
    pipeline_config = load_pipeline_configuration(pipeline_config_path)
    logger = get_logger()

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

    day = datetime.fromisoformat(flow_call_data.pop("day"))
    level = flow_call_data.pop("level")
    code = flow_call_data.pop("code")
    make_image = flow_call_data.pop("make_image")
    make_movie = flow_call_data.pop("make_movie")
    nfiles = len(flow_call_data["file_list"])

    flow_call_data["file_list"] = file_name_to_full_path(flow_call_data["file_list"], pipeline_config["root"])
    flow_call_data["output_movie_dir"] = os.path.join(pipeline_config["ql_root"], flow_call_data["output_movie_dir"])
    flow_call_data.pop("product_code", None)
    flow_call_data["make_image"] = make_image
    flow_call_data["make_movie"] = make_movie

    flow_call_data["file_list"] = file_name_to_full_path(flow_call_data["file_list"], pipeline_config["root"])
    flow_call_data["output_movie_dir"] = os.path.join(pipeline_config["ql_root"], flow_call_data["output_movie_dir"])

    try:
        quicklook_core_flow(**flow_call_data)
    except Exception as e:
        flow_db_entry.state = "failed"
        flow_db_entry.end_time = datetime.now()
        session.commit()
        raise e
    else:
        flow_db_entry.state = "completed"
        flow_db_entry.end_time = datetime.now()

        quicklook_query = (session.query(Quicklook)
                           .filter(Quicklook.day == day)
                           .filter(Quicklook.level == level)
                           .filter(Quicklook.code == code))
        if quicklook_query is None:
            quicklook_query = Quicklook(day=day, level=level, code=code)
            session.add(quicklook_query)
        if make_image:
            quicklook_query.image_made = True
        if make_movie:
            quicklook_query.movie_made = True
            quicklook_query.movie_nfile = nfiles

        session.commit()
