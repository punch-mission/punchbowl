import json
from datetime import UTC, datetime, timedelta

from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE

from punchbowl import __version__
from punchbowl.auto.control.db import File, Flow
from punchbowl.auto.control.processor import generic_process_flow_logic
from punchbowl.auto.control.scheduler import generic_scheduler_flow_logic
from punchbowl.auto.control.util import group_files_by_time
from punchbowl.auto.flows.util import file_name_to_full_path
from punchbowl.level2.flow import level2_core_flow
from punchbowl.util import average_datetime

SCIENCE_POLARIZED_LEVEL1_TYPES = ["PM", "PZ", "PP"]
SCIENCE_CLEAR_LEVEL1_TYPES = ["CR"]


@task(cache_policy=NO_CACHE)
def level2_query_ready_files(session, pipeline_config: dict, reference_time=None, max_n=9e99):
    return _level2_query_ready_files(session, polarized=True, pipeline_config=pipeline_config, max_n=max_n)


@task(cache_policy=NO_CACHE)
def level2_query_ready_clear_files(session, pipeline_config: dict, reference_time=None, max_n=9e99):
    return _level2_query_ready_files(session, polarized=False, pipeline_config=pipeline_config, max_n=max_n)


def _level2_query_ready_files(session, polarized: bool, pipeline_config: dict, max_n=9e99):
    logger = get_run_logger()
    all_ready_files = (session.query(File).filter(File.state == "created")
                       .filter(File.level == "1")
                        # TODO: This line temporarily excludes NFI
                       .filter(File.observatory.in_(["1", "2", "3"]))
                       .filter(File.file_type.in_(
                            SCIENCE_POLARIZED_LEVEL1_TYPES if polarized else SCIENCE_CLEAR_LEVEL1_TYPES))
                       # The ascending sort order is expected by the file grouping code
                       .order_by(File.date_obs.asc()).all())
    logger.info(f"{len(all_ready_files)} ready files")

    if len(all_ready_files) == 0:
        return []

    if polarized:
        grouped_files = group_l2_inputs(all_ready_files)
    else:
        grouped_files = group_files_by_time(all_ready_files, max_duration_seconds=10)

    # Switch to most-recent-first order
    grouped_files = grouped_files[::-1]

    logger.info(f"{len(grouped_files)} sets of grouped files")
    grouped_ready_files = []
    cutoff_time = (pipeline_config["flows"]["level2" if polarized else "level2_clear"]
                   .get("ignore_missing_after_days", None))
    if cutoff_time is not None:
        cutoff_time = datetime.now(tz=UTC) - timedelta(days=cutoff_time)

    for group in grouped_files:
        if len(grouped_ready_files) >= max_n:
            break
        # TODO: This line temporarily excludes NFI
        # group_is_complete = len(group) == (12 if polarized else 4)
        group_is_complete = len(group) == (9 if polarized else 3)
        if group_is_complete:
            grouped_ready_files.append(group)
            continue

        # group[-1] is the newest file by date_obs
        if cutoff_time and group[-1].date_obs.replace(tzinfo=UTC) > cutoff_time:
            # We're still potentially waiting for downlinks
            continue

        # We now have to consider making an incomplete trefoil. We want to look at the L0 files to see if we're still
        # waiting on any L1s. This is especially important when reprocessing. To do that, we need to determine a time
        # range within which to grab L0s

        if polarized:
            # When is the nominal center of this polarized triplet? Remember, we could be missing anything.
            for f in group:
                # If we have a 'Z' image, it's that image's date_obs.
                if f.polarization == "Z":
                    center = f.date_obs
                    break
            else:
                # Grab an arbitrary file, which is either in the first part of the triplet or the last part
                f = group[0]
                # Account for the swapped order of polarization states in NFI/WFI
                if (f.observatory == "4" and f.polarization == "M") or (f.observatory != "4" and f.polarization == "P"):
                    # This image is the start of the triplet (and there's 1 minute between polarization states)
                    center = f.date_obs + timedelta(minutes=1)
                else:
                    # This image is the end of the triplet (and there's 1 minute between polarization states)
                    center = f.date_obs - timedelta(minutes=1)

            # Two minutes from center takes us into the clear exposure/roll on either side of a polarized triplet
            search_width = timedelta(minutes=2)
            search_types = SCIENCE_POLARIZED_LEVEL1_TYPES
        else:
            # So much easier for clears!
            center = group[0].date_obs
            search_width = timedelta(minutes=1)
            search_types = SCIENCE_CLEAR_LEVEL1_TYPES

        # Grab all the L0s that produce inputs for this trefoil
        expected_inputs = (session.query(File)
                                  .filter(File.level == "0")
                                  # TODO: This line temporarily excludes NFI
                                  .filter(File.observatory.in_(["1", "2", "3"]))
                                  .filter(File.file_type.in_(search_types))
                                  .filter(File.date_obs > center - search_width)
                                  .filter(File.date_obs < center + search_width)
                                  .all())
        if len(expected_inputs) == len(group):
            # We have the L1s for all the L0s, and we don't expect new L0s, so let's make an incomplete mosaic
            grouped_ready_files.append(group)
        # Otherwise, we'll pass for now on processing this trefoil
        continue

    logger.info(f"{len(grouped_ready_files)} groups heading out")
    return grouped_ready_files


def group_l2_inputs(files: list[File]) -> list[tuple[File]]:
    """
    Group up L1 inputs into MZP clusters that match in time (i.e. occur sequentially in one image cluster).

    Handles the swapped MZP/PZM orders, handles any combination of missing files, and for each observatory returns only
    complete MZP triplets
    """
    if len(files) == 0:
        return []
    # Sort the files by observatory
    wfi1, wfi2, wfi3, nfi = [], [], [], []
    for file in files:
        match file.observatory:
            case "1":
                wfi1.append(file)
            case "2":
                wfi2.append(file)
            case "3":
                wfi3.append(file)
            case "4":
                nfi.append(file)

    # Build groups per observatory
    wfi1 = group_l2_inputs_single_observatory(wfi1, ["P", "Z", "M"])
    wfi2 = group_l2_inputs_single_observatory(wfi2, ["P", "Z", "M"])
    wfi3 = group_l2_inputs_single_observatory(wfi3, ["P", "Z", "M"])
    nfi = group_l2_inputs_single_observatory(nfi, ["M", "Z", "P"])

    # To group the groups, we'll take the first file of each group, group up the first files, and then fill in those
    # groups with the corresponding second and third files.
    first_files = []
    id_to_group = {}
    for list_of_groups in [wfi1, wfi2, wfi3, nfi]:
        # Only keep full groups (i.e. complete (MZP) triplets)
        list_of_groups[:] = [group for group in list_of_groups if len(group) == 3]
        first_files.extend([g[0] for g in list_of_groups])
        id_to_group.update({g[0].file_id: g for g in list_of_groups})

    if len(first_files) == 0:
        return []

    first_files.sort(key=lambda f: f.date_obs)

    groups = group_files_by_time(first_files, max_duration_seconds=10)

    complete_groups = []
    for group in groups:
        complete_group = []
        for file in group:
            complete_group.extend(id_to_group[file.file_id])
        complete_groups.append(tuple(complete_group))
    return complete_groups


def group_l2_inputs_single_observatory(
        files: list[File], expected_sequence: list[str] | str, max_separation: float=80,
        only_complete=False) -> list[tuple[File]]:
    """
    For a single observatory, groups up L1 inputs into MZP clusters that match in time (i.e. occur sequentially in one
    image cluster).

    Accepts as input the order of P, Z and M, and handles any combination of missing files
    """
    if len(files) == 0:
        return []
    grouped_files = []
    # We'll keep track of where the current group started, and then keep stepping to find the end of this group.
    group_start = 0
    previous_time_stamp = files[0].date_obs.replace(tzinfo=UTC).timestamp()
    file_under_consideration = 0
    previous_code_index = expected_sequence.index(files[file_under_consideration].polarization)
    while True:
        file_under_consideration += 1
        if file_under_consideration == len(files):
            break
        this_tstamp = files[file_under_consideration].date_obs.replace(tzinfo=UTC).timestamp()
        # Check where we are in the expected sequence of polarization states
        this_code_index = expected_sequence.index(files[file_under_consideration].polarization)
        cut_group = False
        if this_code_index <= previous_code_index:
            # We've gone backwards (or at least not forwards) in polarization state, so this must be a new group
            cut_group = True
        else:
            # Based on how far we've advanced in the polarization state sequence, work out the maximum amount of time we
            # can expect to have passed. If more time has passed than that, several images were skipped and we're in the
            # next group.
            allowable_gap = max_separation * (this_code_index - previous_code_index)
            if this_tstamp - previous_time_stamp > allowable_gap:
                cut_group = True
        if cut_group:
            grouped_files.append(tuple(files[group_start:file_under_consideration]))
            group_start = file_under_consideration
        previous_time_stamp = this_tstamp
        previous_code_index = this_code_index
    grouped_files.append(tuple(files[group_start:]))
    if only_complete:
        grouped_files = [group for group in grouped_files if len(group) == len(expected_sequence)]
    return grouped_files


@task(cache_policy=NO_CACHE)
def level2_construct_flow_info(level1_files: list[File], level2_file: File, pipeline_config: dict, session=None, reference_time=None):
    flow_type = "level2_clear" if level1_files[0].file_type == "CR" else "level2"
    state = "planned"
    creation_time = datetime.now()
    priority = pipeline_config["flows"][flow_type]["priority"]["initial"]
    alphas_path = pipeline_config["flows"][flow_type].get("alpha_file_path", None)
    trim_edges_px = pipeline_config["flows"][flow_type].get("trim_edges_px", 0)
    rolloff_width = pipeline_config["flows"][flow_type].get("rolloff_width", .25)
    rolloff_strength = pipeline_config["flows"][flow_type].get("rolloff_strength", 1)
    call_data = json.dumps(
        {
            "data_list": [level1_file.filename() for level1_file in level1_files],
            "voter_filenames": [[] for _ in level1_files],
            "alphas_file": alphas_path,
            "trim_edges_px": trim_edges_px,
            "rolloff_width": rolloff_width,
            "rolloff_strength": rolloff_strength,
        },
    )
    return Flow(
        flow_type=flow_type,
        state=state,
        flow_level="2",
        creation_time=creation_time,
        priority=priority,
        call_data=call_data,
    )


@task
def level2_construct_file_info(level1_files: list[File], pipeline_config: dict, reference_time=None) -> list[File]:
    return [File(
                level="2",
                file_type="CT" if level1_files[0].file_type == "CR" else "PT",
                observatory="M",
                polarization="C" if level1_files[0].file_type == "CR" else "Y",
                file_version=pipeline_config["file_version"],
                software_version=__version__,
                date_obs=average_datetime([f.date_obs for f in level1_files]),
                outlier=any(file.outlier for file in level1_files),
                bad_packets=any(file.bad_packets for file in level1_files),
                state="planned",
            )]


@flow
def level2_scheduler_flow(pipeline_config_path=None, session=None, reference_time=None):
    generic_scheduler_flow_logic(
        level2_query_ready_files,
        level2_construct_file_info,
        level2_construct_flow_info,
        pipeline_config_path,
        reference_time=reference_time,
        session=session,
    )


@flow
def level2_clear_scheduler_flow(pipeline_config_path=None, session=None, reference_time=None):
    generic_scheduler_flow_logic(
        level2_query_ready_clear_files,
        level2_construct_file_info,
        level2_construct_flow_info,
        pipeline_config_path,
        reference_time=reference_time,
        session=session,
    )


def level2_call_data_processor(call_data: dict, pipeline_config, session=None) -> dict:
    call_data["data_list"] = file_name_to_full_path(call_data["data_list"], pipeline_config["root"])
    return call_data

@flow
def level2_process_flow(flow_id: int | list[int], pipeline_config_path=None, session=None):
    generic_process_flow_logic(flow_id, level2_core_flow, pipeline_config_path, session=session,
                               call_data_processor=level2_call_data_processor)


@flow
def level2_clear_process_flow(flow_id: int | list[int], pipeline_config_path=None, session=None):
    generic_process_flow_logic(flow_id, level2_core_flow, pipeline_config_path, session=session,
                               call_data_processor=level2_call_data_processor)
