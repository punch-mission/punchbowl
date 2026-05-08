import os
import json
from datetime import UTC, datetime, timedelta

from dateutil.parser import parse as parse_datetime_str
from prefect import flow, get_run_logger, task

from punchbowl import __version__
from punchbowl.auto.control.db import File, Flow
from punchbowl.auto.control.processor import generic_process_flow_logic
from punchbowl.auto.control.scheduler import generic_scheduler_flow_logic
from punchbowl.auto.control.util import get_database_session, load_pipeline_configuration
from punchbowl.auto.flows.util import file_name_to_full_path
from punchbowl.level3.velocity import track_velocity


@task
def level3_vam_query_ready_files(session, pipeline_config: dict, reference_time: datetime=None, max_n: float=100):
    flow_type = "L3_VAM"
    logger = get_run_logger()
    min_file_count = pipeline_config["flows"]["L3_VAM"]["min_file_count"]

    all_ready_files = (session.query(File)
                   .filter(File.state.in_(["created", "progressed"]))
                   .filter(File.level == "3")
                   .filter(File.file_type == "PT")
                   .filter(File.observatory == "M")
                   .order_by(File.date_obs.desc()).all())

    if len(all_ready_files) == 0:
        return []

    t0 = parse_datetime_str(pipeline_config["flows"][flow_type]["t0"])
    increment = timedelta(minutes=6*60)

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
                        .filter(File.file_type == "PT")
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

    cutoff_time = pipeline_config["flows"][flow_type].get("ignore_missing_after_days", None)
    if cutoff_time is not None:
        cutoff_time = datetime.now(tz=UTC) - timedelta(days=cutoff_time)

    grouped_ready_files = []
    for group in grouped_files:
        if len(grouped_ready_files) >= max_n:
            break

        group_is_complete = len(group) > min_file_count

        if group_is_complete:
            grouped_ready_files.append(group)
            continue

        if cutoff_time and min(f.date_created for f in group).replace(tzinfo=UTC) < cutoff_time:
            # We've waited long enough. Just go ahead and make it.
            grouped_ready_files.append(group)
            continue

    cleaned_ready_groups = []
    for group in grouped_ready_files:
        group = [f for f in group if f.outlier == 0]
        if group:
            cleaned_ready_groups.append(group)

    logger.info(f"{len(cleaned_ready_groups)} groups heading out")
    return cleaned_ready_groups

@task
def level3_vam_construct_flow_info(level3_ptm_files: list[File],
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
            "files": [ptm_file.filename() for ptm_file in level3_ptm_files],
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
        date_beg=min([f.date_obs for f in level3_files if f.outlier == 0]),
        date_end=max([f.date_obs for f in level3_files if f.outlier == 0]),
        state="planned",
        polarization="Y",
        # Outlier images are excluded from VAMs
        outlier=0,
        bad_packets=False,
    )]


@flow
def level3_vam_scheduler_flow(pipeline_config_path=None, session=None):
    generic_scheduler_flow_logic(
        level3_vam_query_ready_files,
        level3_vam_construct_file_info,
        level3_vam_construct_flow_info,
        pipeline_config_path,
        update_input_file_state=False,
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
