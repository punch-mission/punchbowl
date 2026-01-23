import os
import json
import tempfile
from datetime import UTC, datetime, timedelta

from prefect import flow, get_run_logger, task
from prefect.context import get_run_context
from prefect.runtime import flow_run

from punchbowl.data.meta import construct_all_product_codes
from punchbowl.data.punch_io import load_ndcube_from_fits, write_ndcube_to_quicklook, write_quicklook_to_mp4
from punchbowl.pipe.control.db import File, Flow
from punchbowl.pipe.control.util import get_database_session, load_pipeline_configuration, load_quicklook_scaling
from punchbowl.pipe.flows.util import file_name_to_full_path


@task
def visualize_query_ready_files(session, pipeline_config: dict, reference_time: datetime, lookback_hours: float = 24):
    logger = get_run_logger()

    all_ready_files = []
    all_product_codes = []
    levels = ["0", "1", "2", "3", "Q"]
    for level in levels:
        product_codes = construct_all_product_codes(level=level)
        for product_code in product_codes:
            product_ready_files = (session.query(File)
                                    .filter(File.state.in_(["created", "progressed", "quickpunched"]))
                                    .filter(File.date_obs >= (reference_time - timedelta(hours=lookback_hours)))
                                    .filter(File.date_obs <= reference_time)
                                    .filter(File.level == level)
                                    .filter(File.file_type == product_code[0:2])
                                    .filter(File.observatory == product_code[2])
                                    .order_by(File.date_obs.asc()).all())
            logger.info(f"Found {len(product_ready_files)} files to make for {level}_{product_code}")
            all_ready_files.append(list(product_ready_files))
            all_product_codes.append(f"L{level}_{product_code}")

    logger.info(f"{len(all_ready_files)} files will be used for visualization.")
    return all_ready_files, all_product_codes


@task
def visualize_flow_info(input_files: list[File],
                        product_code: str,
                        pipeline_config: dict,
                        reference_time: datetime,
                        session=None,
                        framerate: int = 5,
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
def movie_scheduler_flow(pipeline_config_path=None, session=None, reference_time: datetime | None = None,
                         look_back_hours: float = 24, framerate: int = 5, resolution: int = 1024):
    if session is None:
        session = get_database_session()

    reference_time = reference_time or datetime.now(UTC)

    pipeline_config = load_pipeline_configuration(pipeline_config_path)

    file_lists, product_codes = visualize_query_ready_files(session, pipeline_config, reference_time, look_back_hours)

    for file_list, product_code in zip(file_lists, product_codes):
        if file_list:
            flow = visualize_flow_info(file_list, product_code, pipeline_config, reference_time, session,
                                       framerate=framerate, resolution=resolution)
            session.add(flow)

    session.commit()

def generate_flow_run_name():
    parameters = flow_run.parameters
    code = parameters["product_code"]
    files = parameters["file_list"]
    return f"movie-{code}-len={len(files)}-{datetime.now()}"


@flow(flow_run_name=generate_flow_run_name)
def movie_core_flow(file_list: list, product_code: str, output_movie_dir: str,
                    framerate: int = 5,
                    resolution: int = 1024,
                    ffmpeg_cmd: str = "ffmpeg") -> None:
    tempdir = tempfile.TemporaryDirectory()

    annotation = "{OBSRVTRY} - {TYPECODE}{OBSCODE} - {DATE-OBS} - polarizer: {POLAR} deg - exptime: {EXPTIME} secs - LEDPLSN: {LEDPLSN}"
    written_list = []
    if file_list:
        for i, cube_file in enumerate(file_list):
            cube = load_ndcube_from_fits(cube_file)

            if i == 0:
                obs_start = cube.meta.datetime
            if i == len(file_list)-1:
                obs_end = cube.meta.datetime

            img_file = os.path.join(tempdir.name, os.path.splitext(os.path.basename(cube_file))[0] + ".jp2")

            written_list.append(img_file)

            vmin, vmax = load_quicklook_scaling(level=cube.meta["LEVEL"].value, product=cube.meta["TYPECODE"].value, obscode=cube.meta["OBSCODE"].value)

            if cube.meta["LEVEL"].value == 0 and cube.meta["ISSQRT"].value == 0:
                vmin = vmin**2
                vmax = vmax**2

            write_ndcube_to_quicklook(cube, filename=img_file, annotation=annotation, vmin=vmin, vmax=vmax)

        out_filename = os.path.join(output_movie_dir,
                                    f"{product_code}_{obs_start.isoformat()}-{obs_end.isoformat()}.mp4")
        os.makedirs(os.path.dirname(out_filename), exist_ok=True)
        write_quicklook_to_mp4(files=written_list, filename=out_filename,
                               ffmpeg_cmd=ffmpeg_cmd,
                               framerate=framerate, resolution=resolution)

        tempdir.cleanup()


@flow
def movie_process_flow(flow_id: int, pipeline_config_path=None, session=None):
    if session is None:
        session = get_database_session()
    pipeline_config = load_pipeline_configuration(pipeline_config_path)
    logger = get_run_logger()

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

    flow_call_data["file_list"] = file_name_to_full_path(flow_call_data["file_list"], pipeline_config["root"])
    flow_call_data["output_movie_dir"] = os.path.join(pipeline_config["root"], flow_call_data["output_movie_dir"])

    try:
        movie_core_flow(**flow_call_data)
    except Exception as e:
        flow_db_entry.state = "failed"
        flow_db_entry.end_time = datetime.now()
        session.commit()
        raise e
    else:
        flow_db_entry.state = "completed"
        flow_db_entry.end_time = datetime.now()
        # Note: the file_db_entry gets updated above in the writing step because it could be created or blank
        session.commit()
