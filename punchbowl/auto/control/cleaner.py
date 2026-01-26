import os
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from prefect.client.orchestration import get_client
from prefect.client.schemas.filters import (
    FlowRunFilter,
    FlowRunFilterStartTime,
    FlowRunFilterState,
    FlowRunFilterStateType,
)
from prefect.client.schemas.objects import StateType
from sqlalchemy.orm import aliased

from punchbowl.auto.control.db import File, FileRelationship, Flow
from punchbowl.auto.control.util import get_database_session, load_pipeline_configuration


@flow
async def cleaner(pipeline_config_path: str, session=None):
    logger = get_run_logger()

    pipeline_config = load_pipeline_configuration(pipeline_config_path)
    if session is None:
        session = get_database_session()

    reset_revivable_flows(logger, session, pipeline_config)

    # because flows in the launched state aren't running in Prefect yet, we don't update them there
    await fail_stuck_flows(logger, session, pipeline_config, "launched", update_prefect=False)

    # running flows are both in Prefect and in our punchpipe database, so we have to cancel them both places
    await fail_stuck_flows(logger, session, pipeline_config, "running", update_prefect=True)

@task(cache_policy=NO_CACHE)
def reset_revivable_flows(logger, session, pipeline_config):
    # Note: I thought about adding a maximum here, but this flow takes only 5 seconds to revive 10,000 L1 flows, so I
    # think we're good.
    child = aliased(File)
    parent = aliased(File)
    results = (session.query(Flow, child, FileRelationship, parent)
               # isouter sets a left outer join---we'll get back flows that don't have a matching child file,
               # but not files without a matching flow
               .join(child, Flow.flow_id == child.processing_flow, isouter=True)
               .join(FileRelationship, child.file_id == FileRelationship.child, isouter=True)
               .join(parent, parent.file_id == FileRelationship.parent, isouter=True)
               .where(Flow.state == "revivable")
               ).all()

    # This one loops differently than the others, because we need to track the child that's being deleted to know how
    # to reset the parent.
    unique_parents = set()
    for processing_flow, _, _, parent in results:
        if parent is None:
            continue
        if processing_flow.flow_type not in ("construct_stray_light",
                                             "construct_dynamic_stray_light"
                                             "construct_f_corona_background",
                                             "construct_starfield_background",
                                             "levelq_CFM"):
            parent.state = "created"
            unique_parents.add(parent.file_id)
    logger.info(f"Reset {len(unique_parents)} parent files")

    unique_children = {child for _, child, _, _ in results if child is not None}
    root_path = Path(pipeline_config["root"])
    for child in unique_children:
        output_path = Path(child.directory(pipeline_config["root"])) / child.filename()
        if output_path.exists():
            os.remove(output_path)
        sha_path = str(output_path) + ".sha"
        if os.path.exists(sha_path):
            os.remove(sha_path)
        jp2_path = output_path.with_suffix(".jp2")
        if jp2_path.exists():
            os.remove(jp2_path)
        # Iteratively remove parent directories if they're empty. output_path.parents gives the file's parent dir,
        # then that dir's parent, then that dir's parent...
        for parent_dir in output_path.parents:
            if not parent_dir.exists():
                break
            if len(os.listdir(parent_dir)):
                break
            if parent_dir == root_path:
                break
            parent_dir.rmdir()
        session.delete(child)
    logger.info(f"Deleted {len(unique_children)} child files")

    # Every FileRelationship item is unique
    n_cleared = 0
    for _, _, relationship, _ in results:
        if relationship is not None:
            session.delete(relationship)
            n_cleared += 1
    logger.info(f"Cleared {n_cleared} file relationships")

    unique_flows = {flow for flow, _, _, _ in results}
    for f in unique_flows:
        session.delete(f)
    logger.info(f"Deleted {len(unique_flows)} flows")

    session.commit()
    if unique_flows:
        logger.info(f"Processed {len(unique_flows)} revivable flows")


@task(cache_policy=NO_CACHE)
async def cancel_running_prefect_flows_before_cutoff(
        cutoff: datetime,
        batch_size: int = 100,
):
    """Cancels flows that started running before a cutoff time."""
    logger = get_run_logger()

    async with get_client() as client:
        flow_run_filter = FlowRunFilter(
            start_time=FlowRunFilterStartTime(before_=cutoff),
            state=FlowRunFilterState(
                type=FlowRunFilterStateType(
                    any_=[StateType.RUNNING],
                ),
            ),
        )

        # Get flow runs to delete
        flow_runs = await client.read_flow_runs(
            flow_run_filter=flow_run_filter,
            limit=batch_size,
        )

        if not flow_runs:
            logger.info("No flows to delete")
            return

        n_cancelled = 0
        while flow_runs:
            # Cancel each flow run through the API
            for i, flow_run in enumerate(flow_runs):
                # First we send a cancel signal. If the underlying is process actually is still running, we don't
                # want to leave it running unmonitored. (Ideally we wouldn't be cancelling it at all, but failing
                # that, this is the next best thing.) There doesn't appear to be a way to cancel flows from the
                # python client, so we have to roll up our sleeves. There also doesn't appear to be a way to get the
                # PID of the underlying flow so we can kill it ourselves.

                logger.info(f"Cancelling flow {flow_run.name}")
                subprocess.run(["prefect", "flow-run", "cancel", str(flow_run.id)], check=False)
                n_cancelled += 1
                # Rate limiting - adjust based on your API capacity
                if i % 10 == 0:
                    await asyncio.sleep(0.5)

            logger.info(f"Cancelled a batch of {len(flow_runs)} flow runs (total: {n_cancelled})")

            # Delay between batches to avoid overwhelming the API
            await asyncio.sleep(1.0)

            # Get next batch
            flow_runs = await client.read_flow_runs(
                flow_run_filter=flow_run_filter,
                limit=batch_size,
            )

        # Give time for Prefect to kill the processes
        logger.info("Giving time for cancellations to happen...")
        await asyncio.sleep(30)

        # *Now* we can delete them. If they cancelled properly there's probably no need, but anecdotally if you try
        # to cancel a flow when the underlying process isn't there (e.g. if you restarted the pipeline, and this
        # flow is from before the restart), the cancellation never completes, and we need to make sure concurrency slots
        # get freed. So we re-check for stuck running flows and delete those that failed to cancel.

        flow_runs = await client.read_flow_runs(
            flow_run_filter=flow_run_filter,
            limit=batch_size,
        )
        deleted_total = 0
        failed_deletes = []
        while flow_runs:
            for i, flow_run in enumerate(flow_runs):
                try:
                    logger.info(f"Deleting {flow_run.name} from Prefect")
                    await client.delete_flow_run(flow_run.id)
                    deleted_total += 1
                except Exception as e:
                    logger.warning(f"Failed to delete flow run {flow_run.id}: {e}")
                    failed_deletes.append(flow_run.id)

                # Rate limiting - adjust based on your API capacity
                if i % 10 == 0:
                    await asyncio.sleep(0.5)

            logger.info(f"Deleted a batch of {len(flow_runs)} flow runs (total: {deleted_total})")

            # Delay between batches to avoid overwhelming the API
            await asyncio.sleep(1.0)

            # Get next batch
            flow_runs = await client.read_flow_runs(
                flow_run_filter=flow_run_filter,
                limit=batch_size,
            )

        logger.info(f"Deleted {deleted_total} flow runs")
        if failed_deletes:
            logger.warning(f"Failed to delete {len(failed_deletes)} flow runs")


@task(cache_policy=NO_CACHE)
async def fail_stuck_flows(logger, session, pipeline_config, state, update_prefect=False):
    amount_of_patience = pipeline_config["control"]["cleaner"].get(f"fail_{state}_flows_after_minutes", -1)
    if amount_of_patience < 0:
        logger.warning(f"There is no fail_{state}_flows_after_minutes option in the config, so ending without checking.")
        return

    # First, we get the flows that are stuck. This should happen before the flows are killed, in case the flow's
    # on_failure hook is able to change the flow state.
    cutoff = datetime.now() - timedelta(minutes=amount_of_patience)
    stucks = (session.query(Flow)
              .where(Flow.state == state)
              .where(Flow.launch_time < cutoff)
              ).all()

    # Next we try to kill any of the stuck flows that are still running, so they release any DB locks they hold.
    # we clean the prefect database even if our database returned no stucks because they might have somehow gotten
    # out of sync. we want to clean that up too
    if update_prefect:
        # The postgres database has timezone-aware timestamps
        local_timezone = datetime.now().astimezone().tzinfo
        cutoff = cutoff.replace(tzinfo=local_timezone)
        await cancel_running_prefect_flows_before_cutoff(cutoff)

    # With the locks hopefully released, we can change the DB states
    if len(stucks):
        for stuck in stucks:
            stuck.state = "timed_out"
            logger.info(f"Timing out flow {stuck.flow_id} {stuck.flow_run_name}")
        # Mark the output files as timed_out
        files = session.query(File).where(File.processing_flow.in_([s.flow_id for s in stucks])).all()
        for file in files:
            file.state = "timed_out"

        session.commit()
        logger.info(f"Failed {len(stucks)} flows that have been "
                    f"in a '{state}' state for {amount_of_patience} minutes from punchpipe database")
