import os
import random
from datetime import datetime, timedelta

from punchbowl.auto.control.db import File
from punchbowl.auto.control.util import group_files_by_time


def file_name_to_full_path(file_name: str | None | list[str | None], root_dir: str) -> str | None | list[str | None]:
    if isinstance(file_name, list):
        return [file_name_to_full_path(f, root_dir) for f in file_name]
    if file_name is None:
        return None
    level = file_name.split("_")[1][1]
    code = file_name.split("_")[2][:2]
    obs = file_name.split("_")[2][-1]
    version = file_name.split("_")[-1].split(".")[0][1:]
    date = datetime.strptime(file_name.split("_")[3], "%Y%m%d%H%M%S")

    file = File(level=level, file_type=code, observatory=obs, file_version=version, software_version="", date_obs=date,
                state="")

    return os.path.join(file.directory(root_dir), file_name)


def summarize_files_missing_cal_files(files: list[File]):
    if len(files) < 1000:
        summary = ", ".join([f.filename() for f in files])
    else:
        dates = [f.date_obs for f in files]
        types = {f.file_type + f.observatory for f in files}
        summary = (f"{len(files)} files, of types {sorted(types)}, with date-obs ranging from "
                   f"{min(dates).isoformat()} to {max(dates).isoformat()}")
    return summary


def pick_closest_file(files: list[File], target_file: File, max_difference: timedelta | None = None) -> File | None:
    """
    Pick from a list of files the one closest to a target file.

    Parameters
    ----------
    files : list[File]
        The files to choose from
    target_file : File
        The file from which "closest to" is measured
    max_difference : timedelta | None
        The maximum allowed difference between the closest file and the target file. If set and the closest file exceeds
        this separation, None will be returned.

    Returns
    -------
    File | None
        The closest file
    """
    if not files:
        return None
    files = sorted(files, key=lambda f: abs(f.date_obs - target_file.date_obs))
    closest = files[0]
    if max_difference is not None and abs(closest.date_obs - target_file.date_obs) > max_difference:
        return None
    return closest


def get_max_flows_to_schedule(pipeline_config: dict, flow_type: str) -> int:
    """
    Work out the maximum number of flows to schedule for a flow type.

    Will use the first of these to exist:
        - The flow's "scheduling_limit" value
        - The flow's "concurrency_limit" value
        - The global "scheduler/max_start" value

    Parameters
    ----------
    pipeline_config : dict
        The pipeline configuration
    flow_type : str
        The flow type to handle

    Returns
    -------
    n_flows : int
        The maximum number of flows to have scheduled for this flow type.

    """
    if 'scheduling_limit' in pipeline_config["flows"][flow_type]:
        return pipeline_config["flows"][flow_type]['scheduling_limit']
    elif 'concurrency_limit' in pipeline_config["flows"][flow_type]:
        return pipeline_config["flows"][flow_type]['concurrency_limit']
    else:
        return pipeline_config['scheduler']['max_start']


def apply_stride(files: list[File], stride: int, actual_image_cadence_minutes: float) -> list[File]:
    """
    Apply a stride in a dynamic way.

    When we're making a background product and we need a large time window, but using every image in that window is
    overkill, we want to apply a stride and pick every Nth image in that time window. But if, for example,
    part of that time window includes eclipse season, and so good files are less dense in some parts of the window,
    we don't need to stride as much there. If only every Nth image exists and is good, we don't need to apply another
    "take every Nth" filter---we can just take every existing file for that period.

    This function takes the files in a given time range, groups them into [N * nominal image cadence - epsilon]
    groups, and picks one image from every group. This way, portions of the larger window that have full image
    coverage are down-sampled, but portions that have a lower image density are handled appropriately, and our final
    image selection is more evenly spaced through the time window.

    Parameters
    ----------
    files : list[File]
        The files to down-select
    stride : int
        The down-selection factor for periods where images are at nominal cadence. This is the "we want one out of every
        N images" number.
    actual_image_cadence_minutes : float
        The nominal time between images

    Returns
    -------
    list[File]
        The selected files.
    """
    files = sorted(files, key=lambda f: f.date_obs)
    # We need to pull in the end of the group, or else we'll have the image that should start the next group
    epsilon = 20
    groups = group_files_by_time(files, max_duration_seconds=actual_image_cadence_minutes * 60 * stride - epsilon)
    selected_files = []
    random.seed(1)
    for group in groups:
        # Apply a stride that doesn't phase weirdly with where we are in a roll position
        selected_files.append(random.choice(group))
    return selected_files
