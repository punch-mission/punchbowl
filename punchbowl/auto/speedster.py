import os
import time
import argparse
import warnings
import traceback
import multiprocessing
from datetime import datetime
from collections import defaultdict

import prefect.exceptions
import yaml
from prefect.logging import disable_run_logger
from sqlalchemy import update
from tqdm.auto import tqdm
from yaml.loader import FullLoader

from punchbowl.auto.cli import find_flow
from punchbowl.auto.control.db import Flow
from punchbowl.auto.control.util import get_database_session


def load_pipeline_configuration(path: str = None) -> dict:
    with open(path) as f:
        config = yaml.load(f, Loader=FullLoader)
    # TODO: add validation
    return config


def load_enabled_flows(pipeline_config):
    enabled_flows = []
    for flow_type in pipeline_config["flows"]:
        if pipeline_config["flows"][flow_type].get("enabled", True) == "speedy":
            enabled_flows.append(flow_type)
    return enabled_flows


def gather_planned_flows(session, enabled_flows, max_n=None):
    flows = (session.query(Flow)
             .where(Flow.state == "planned")
             .where(Flow.flow_type.in_(enabled_flows))
             .order_by(Flow.is_backprocessing.asc(), Flow.priority.desc(), Flow.creation_time.asc())
             .limit(max_n).all())
    count_per_type = defaultdict(lambda: 0)
    flow_ids = []
    types = []
    for flow in flows:
        types.append(flow.flow_type)
        count_per_type[flow.flow_type] += 1
        flow_ids.append(flow.flow_id)

    return flow_ids, types, count_per_type


def worker_init(config_path):
    global session, flow_type_to_runner, path_to_config
    with disable_run_logger(), warnings.catch_warnings():
        # Otherwise warning spam will hide any progress messages
        warnings.simplefilter("ignore")
        session = get_database_session()
    flow_type_to_runner = dict()
    path_to_config = config_path


def worker_run_flow(inputs):
    flow_id, flow_type, delay = inputs
    global flow_type_to_runner, session, path_to_config
    if flow_type not in flow_type_to_runner:
        runner = find_flow(flow_type + "_process_flow").fn
        flow_type_to_runner[flow_type] = runner
    else:
        runner = flow_type_to_runner[flow_type]

    session.execute(update(Flow).where(Flow.flow_id == flow_id).values(
            state="launched", flow_run_name="speedster", launch_time=datetime.now()))

    with disable_run_logger(), warnings.catch_warnings():
        # Otherwise warning spam will hide any progress messages
        warnings.simplefilter("ignore")
        try:
            time.sleep(delay)
            runner(flow_id, path_to_config, session)
        except (KeyboardInterrupt, prefect.exceptions.TerminationSignal):
            session.execute(
                update(Flow).where(Flow.flow_id == flow_id).values(state="revivable"))
            session.commit()
            print(f"Keyboard interrupt in flow {flow_id}; marked as revivable")
        except: # noqa: E722
            print(f"Exception in flow {flow_id}")
            traceback.print_exc()


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    parser = argparse.ArgumentParser(prog="speedster")
    parser.add_argument("config", type=str, help="Path to config.")
    parser.add_argument("-f", "--flows-per-batch", type=int, help="Max number of flows per batch.")
    parser.add_argument("-b", "--n-batches", type=int, help="Number of batches.")
    parser.add_argument("-w", "--n-workers", type=int, help="Number of workers")
    args = parser.parse_args()
    config_path = args.config

    session = get_database_session(engine_kwargs=dict(isolation_level="READ COMMITTED"))

    if args.n_workers is None:
        args.n_workers = os.cpu_count()

    if args.flows_per_batch is None:
        n_cores = args.n_workers
    else:
        n_cores = min(args.n_workers, args.flows_per_batch)

    n_batches_run = 0
    with multiprocessing.Pool(n_cores, initializer=worker_init, initargs=(config_path,)) as p:
        print("Beginning fetch-run loop; press Ctrl-C to exit and allow time for cleanup")
        if args.flows_per_batch:
            print(f"Will cap at {args.flows_per_batch} flows per batch")
        if args.n_batches:
            print(f"Will stop after {args.n_batches} batches")
        while True:
            pipeline_config = load_pipeline_configuration(config_path)
            enabled_flows = load_enabled_flows(pipeline_config)

            batch_of_flows, batch_types, count_per_type = gather_planned_flows(
                    session, enabled_flows, args.flows_per_batch)

            if len(batch_of_flows) == 0:
                print("No pending flows found---will wait two minutes and try again")
                try:
                    time.sleep(60*2)
                except KeyboardInterrupt:
                    break
            else:
                print("Batch contents: ", end="")
                count_report = []
                for type in sorted(count_per_type.keys()):
                    print(f"{count_per_type[type]} of {type}, ", end="")
                print()
                with tqdm(total=len(batch_of_flows)) as pbar:
                    # Stagger the launches which may give less DB and IO contention
                    delays = [i / 6 if i < n_cores else 0 for i in range(len(batch_of_flows))]
                    try:
                        for _ in p.imap_unordered(worker_run_flow, zip(batch_of_flows, batch_types, delays)):
                            pbar.update()
                    except KeyboardInterrupt:
                        print("Halting")
                        break
            n_batches_run += 1
            if args.n_batches and n_batches_run >= args.n_batches:
                break
