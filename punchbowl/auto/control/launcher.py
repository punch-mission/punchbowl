import asyncio
from math import ceil
from random import shuffle
from datetime import datetime, timedelta
from collections import defaultdict

from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from prefect.client.orchestration import get_client
from prefect.variables import Variable
from sqlalchemy import and_, func, select, update
from sqlalchemy.orm import Session

from punchbowl.auto.control.db import File, Flow
from punchbowl.auto.control.util import batched, get_database_session, load_pipeline_configuration


@task(cache_policy=NO_CACHE)
def gather_planned_flows(session, weight_to_launch, max_flows_to_launch, flow_weights, flow_enabled, flow_batch_sizes):
    # We'll have to grab a bunch of possible flows to launch from the DB, and then on our end apply the weights and the
    # maximum-weight limit. But we can use the smallest weight to set an upper bound on how many launchable flows to
    # retrieve.
    enabled_flows = [flow for flow, enabled in flow_enabled.items() if enabled]
    if enabled_flows:
        max_to_select = weight_to_launch / min([flow_weights[k] for k in enabled_flows])
    else:
        max_to_select = 0
    flows = (session.query(Flow)
                   .where(Flow.state == "planned")
                   .where(Flow.flow_type.in_(enabled_flows))
                   .order_by(Flow.is_backprocessing.asc(), Flow.priority.desc(), Flow.creation_time.asc())
                   .limit(max_to_select).all())
    selected_flows = []
    selected_weight = 0
    selected_number = 0
    count_per_type = defaultdict(lambda: 0)
    while selected_weight < weight_to_launch and selected_number < max_flows_to_launch and len(flows):
        flow = flows.pop(0)
        if not flow_enabled[flow.flow_type]:
            continue
        selected_flows.append(flow)
        selected_weight += flow_weights[flow.flow_type]
        selected_number += 1 / flow_batch_sizes[flow.flow_type]
        count_per_type[flow.flow_type] += 1

    select_flow_ids = [flow.flow_id for flow in selected_flows]
    output_files = session.query(File).where(File.processing_flow.in_(select_flow_ids)).all()
    tags_by_flow = {}
    for flow in selected_flows:
        tags = set()
        for output_file in output_files:
            if output_file.processing_flow == flow.flow_id:
                tags.add(output_file.file_type + output_file.observatory)
        tags_by_flow[flow.flow_id] = sorted(tags)

    number_of_flows = len(selected_flows)
    batched_flows = []
    for flow_type, batch_size in flow_batch_sizes.items():
        if batch_size <= 1:
            continue
        these_flows, other_flows = [], []
        for flow in selected_flows:
            if flow.flow_type == flow_type:
                these_flows.append(flow)
            else:
                other_flows.append(flow)
        batched_flows.extend(batched(these_flows, batch_size))
        selected_flows = other_flows
    selected_flows = [[f] for f in selected_flows] + batched_flows

    return selected_flows, tags_by_flow, selected_weight, number_of_flows, count_per_type


@task(cache_policy=NO_CACHE)
def count_flows(session, weights):
    n_planned, n_running = 0, 0
    weight_planned, weight_running = 0, 0
    rows = session.execute(
        select(Flow.state, Flow.flow_type, func.count())
        .select_from(Flow)
        .where(Flow.state.in_(("planned", "running", "launched")))
        .group_by(Flow.state, Flow.flow_type),
    ).all()
    # We won't get results for states that aren't actually in the database, so we have to inspect the returned rows
    for state, flow_type, count in rows:
        if state == "planned":
            n_planned += count
            weight_planned += count * weights[flow_type]
        else:
            n_running += count
            weight_running += count * weights[flow_type]
    return n_running, n_planned, weight_planned, weight_running


@task(cache_policy=NO_CACHE)
def escalate_long_waiting_flows(session, pipeline_config):
    for flow_type in pipeline_config["flows"]:
        for max_seconds_waiting, escalated_priority in zip(
            pipeline_config["flows"][flow_type]["priority"]["seconds"],
            pipeline_config["flows"][flow_type]["priority"]["escalation"],
        ):
            since = datetime.now() - timedelta(seconds=max_seconds_waiting)
            session.query(Flow).where(
                and_(Flow.priority < escalated_priority,
                     Flow.state == "planned",
                     Flow.creation_time < since,
                     Flow.flow_type == flow_type),
            ).update({"priority": escalated_priority})
            # Commit after every update to try to avoid deadlocks. I think the problem scenario is (1) we've updated
            # a flow's priority, so we have that row locked until we commit, (2) the flow gets launched, so another
            # process wants to update that flow record and so has a pending transaction on that row, and (3) we
            # continue going through our loops, where each update call has to lock the whole table so it can scan
            # every record, and so it has to wait for the other process, which has its pending lock/update
            session.commit()


def determine_launchable_flow_count(weight_planned, weight_running, max_weight_running, max_weight_to_launch,
                                    max_flows_to_launch):
    logger = get_run_logger()
    amount_to_launch = max_weight_running - weight_running
    logger.info(f"Total weight {amount_to_launch:.2f} can be launched at this time.")

    amount_to_launch = min(amount_to_launch, max_weight_to_launch)
    amount_to_launch = max(0, amount_to_launch)
    logger.info(f"Will launch up to {amount_to_launch:.2f} weight and {max_flows_to_launch} flows")

    return min(amount_to_launch, weight_planned), max_flows_to_launch


@task(cache_policy=NO_CACHE)
async def launch_ready_flows(session: Session, flow_info: list[list[Flow]], tags_by_flow: dict[int, str], pipeline_config: dict) -> None:
    """
    Given a list of ready-to-launch flow_ids, this task creates flow runs in Prefect for them.
    These flow runs are automatically marked as scheduled in Prefect and will be picked up by a work queue and
    agent as soon as possible.

    Parameters
    ----------
    session : sqlalchemy.orm.session.Session
        A SQLAlchemy session for database interactions
    flow_info : List[int]
        A list of flow IDs from the punchpipe database identifying which flows to launch

    Returns
    -------
    A list of responses from Prefect about the flow runs that were created

    """
    if not len(flow_info):
        return
    logger = get_run_logger()
    # gather the flow information for launching

    # If we don't shuffle, flows will be sorted by priority which may implicitly be a sort by flow type. This could
    # mean we launch all the quick flows at once and then later all the slow flows at once, but we'll get better
    # performance overall by mixing the fast and slow flows since the total CPU demand will be more uniform through this
    # scheduling window.
    shuffle(flow_info)

    async with get_client() as client:
        # determine the deployment ids for each kind of flow
        deployments = await client.read_deployments()
        deployment_ids = {d.name: d.id for d in deployments}

        # We want to stagger launches through a time window. If our configured time window is 5 minutes, we'll use 4
        # full minutes, plus a portion of the fifth, aiming to end after 4m35s to leave margin so the flow is fully
        # finished after 5 minutes.
        # First we work out the remainder part, figuring out where we are relative to the 35th second of the current
        # minute.
        total_delay_time = 35 - datetime.now().second
        total_delay_time = max(0, total_delay_time)
        total_delay_time += (pipeline_config["control"]["launcher"]["launch_time_window_minutes"] - 1) * 60
        # Launch a chunk every 10 seconds through this window
        n_chunks = total_delay_time // 10
        n_chunks = max(n_chunks, 1)
        chunk_size = ceil(len(flow_info) / n_chunks)
        logger.info(f"Total delay time: {total_delay_time}")
        if chunk_size >= len(flow_info):
            delay_time = 0
        else:
            delay_time = total_delay_time / (n_chunks - 1)
        awaitables = []
        responses = []
        all_chunks = list(batched(flow_info, chunk_size))
        for chunk_number, chunk in enumerate(all_chunks):
            start = datetime.now().timestamp()

            for batch in chunk:
                for flow in batch:
                    flow.state = "launched"
                    flow.launch_time = datetime.now()
            session.commit()

            # Launch the chunk
            n_actual_flows = 0
            for batch in chunk:
                flow_ids = [flow.flow_id for flow in batch]
                flow_types = set(flow.flow_type for flow in batch)
                assert len(flow_types) == 1
                unique_tags = set()
                for flow in batch:
                    unique_tags.update(tags_by_flow[flow.flow_id])
                unique_tags = list(unique_tags)
                if len(flow_ids) > 1:
                    unique_tags.append("batch")
                this_deployment_id = deployment_ids[batch[0].flow_type + "_process_flow"]
                parameters = {"flow_id": flow_ids[0] if len(flow_ids) == 1 else flow_ids}
                awaitables.append(client.create_flow_run_from_deployment(
                    this_deployment_id, parameters=parameters, tags=unique_tags),
                )
                n_actual_flows += len(flow_ids)

            responses.extend(await asyncio.gather(*awaitables))
            awaitables = []
            logger.info(f"Chunk {chunk_number}/{len(all_chunks)} sent, containing {n_actual_flows} flows "
                        f"in {len(chunk)} batches")
            if delay_time:
                # Stagger the launches
                await asyncio.sleep(delay_time - (datetime.now().timestamp() - start))
        # TODO This doesn't seem to be an effective way to check for a failed flow submission, but we should
        # do something like this that works
        ok_responses = [r for r in responses if r.name not in [None, ""] and r.state_name == "Scheduled"]
        bad_responses = [r for r in responses if r not in ok_responses]

        if bad_responses:
            bad_flow_ids = []
            for r in bad_responses:
                flow_id = r.parameters["flow_id"]
                if isinstance(flow_id, list):
                    bad_flow_ids.extend(flow_id)
                else:
                    bad_flow_ids.append(flow_id)
            session.execute(
                update(Flow)
                .where(Flow.state == "launched")
                .where(Flow.flow_id.in_(bad_flow_ids))
                .values(state="planned"),
            )
            session.commit()
            for r in bad_responses:
                logger.warning(f"Got bad response {r!r}")


def load_flow_data(pipeline_config):
    flow_weights = dict()
    flow_enabled = dict()
    flow_batch_size = dict()
    for flow_type in pipeline_config["flows"]:
        flow_enabled[flow_type] = pipeline_config["flows"][flow_type].get("enabled", True) is True
        flow_weights[flow_type] = pipeline_config["flows"][flow_type].get("launch_weight", 1)
        flow_batch_size[flow_type] = pipeline_config["flows"][flow_type].get("batch_size", 1)
    return flow_weights, flow_enabled, flow_batch_size


@flow
async def launcher(pipeline_config_path=None):
    """
    The main launcher flow for Prefect, responsible for identifying flows, based on priority,
        that are ready to run and creating flow runs for them. It also escalates long-waiting flows' priorities.

    See EM 41 or the internal requirements document for more details

    Returns
    -------
    Nothing

    """
    logger = get_run_logger()

    if pipeline_config_path is None:
        pipeline_config_path = await Variable.get("punchpipe_config", "punchpipe_config.yaml")
    pipeline_config = load_pipeline_configuration(pipeline_config_path)
    flow_weights, flow_enabled, flow_batch_sizes = load_flow_data(pipeline_config)
    logger.info(f"Enabled flows: {', '.join([flow for flow, enabled in flow_enabled.items() if enabled])}")

    logger.info("Establishing database connection")
    session = get_database_session()

    escalate_long_waiting_flows(session, pipeline_config)

    # Perform the launcher flow responsibilities
    num_running_flows, num_planned_flows, weight_planned, weight_running = count_flows(session, flow_weights)
    logger.info(f"There are {num_running_flows} flows running right now (weight {weight_running:.2f}) and {num_planned_flows} planned flows (weight {weight_planned:.2f}).")
    max_weight_running = pipeline_config["control"]["launcher"]["max_weight_running"]
    max_weight_to_launch = pipeline_config["control"]["launcher"]["max_weight_to_launch_at_once"]
    max_flows_to_launch = pipeline_config["control"]["launcher"]["max_flows_to_launch_at_once"]

    weight_to_launch, max_flows_to_launch = determine_launchable_flow_count(
        weight_planned, weight_running, max_weight_running, max_weight_to_launch, max_flows_to_launch)

    flows_to_launch, tags_by_flow, selected_weight, number_of_flows, counts_per_type = gather_planned_flows(
        session, weight_to_launch, max_flows_to_launch, flow_weights, flow_enabled, flow_batch_sizes)
    ids = [[flow.flow_id for flow in batch] for batch in flows_to_launch]
    logger.info(f"{number_of_flows} flows (weight {selected_weight:.2f}) with IDs of {ids} will be launched.")
    counts = [f"{counts_per_type[type]} {type}" for type in sorted(counts_per_type.keys())]
    if counts:
        logger.info("This consists of " + ", ".join(counts))
    await launch_ready_flows(session, flows_to_launch, tags_by_flow, pipeline_config)
    logger.info("Launcher flow exit.")
