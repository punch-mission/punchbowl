import os
import time
import inspect
import argparse
import traceback
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from importlib import import_module

import pandas as pd
from prefect import Flow, get_client, serve
from prefect.client.schemas.objects import ConcurrencyLimitConfig, ConcurrencyLimitStrategy
from prefect.variables import Variable

from punchbowl.auto.control.util import load_pipeline_configuration

THIS_DIR = os.path.dirname(__file__)

def main():
    """Run the PUNCH automated pipeline"""
    parser = argparse.ArgumentParser(prog="punchpipe")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the pipeline.")
    serve_control_parser = subparsers.add_parser("serve-control", help="Serve the control flows.")
    serve_data_parser = subparsers.add_parser("serve-data", help="Serve the data-processing flows.")

    run_parser.add_argument("config", type=str, help="Path to config.")
    run_parser.add_argument("--launch-prefect", action="store_true", help="Launch the prefect server")
    run_parser.add_argument("--no-dask-cluster", action="store_true", help="Skip launching the dask cluster")
    serve_control_parser.add_argument("config", type=str, help="Path to config.")
    serve_data_parser.add_argument("config", type=str, help="Path to config.")
    args = parser.parse_args()

    if args.command == "run":
        run(args.config, args.launch_prefect, not args.no_dask_cluster)
    elif args.command == "serve-data":
        run_data(args.config)
    elif args.command == "serve-control":
        run_control(args.config)
    else:
        parser.print_help()

def find_flow(target_flow, subpackage="flows") -> Flow:
    for filename in os.listdir(os.path.join(THIS_DIR, subpackage)):
        if filename.endswith(".py"):
            module_name = f"punchpipe.{subpackage}."  + os.path.splitext(filename)[0]
            module = import_module(module_name)
            for name, obj in inspect.getmembers(module):
                if name == target_flow:
                    return obj
    raise RuntimeError(f"No flow found for {target_flow}")

def construct_flows_to_serve(configuration_path, include_data=True, include_control=True):
    config = load_pipeline_configuration(configuration_path)

    # create each kind of flow. add both the scheduler and process flow variant of it.
    flows_to_serve = []
    if include_data:
        for flow_name in config["flows"]:
            # first we deploy the scheduler flow
            specific_name = flow_name + "_scheduler_flow"
            specific_tags = config["flows"][flow_name].get("tags", [])
            specific_description = config["flows"][flow_name].get("description", "")
            flow_function = find_flow(specific_name)
            flow_deployment = flow_function.to_deployment(
                name=specific_name,
                description="Scheduler: " + specific_description,
                tags = ["scheduler"] + specific_tags,
                cron=config["flows"][flow_name].get("schedule", None),
                concurrency_limit=ConcurrencyLimitConfig(
                    limit=1,
                    collision_strategy=ConcurrencyLimitStrategy.CANCEL_NEW,
                ),
                parameters={"pipeline_config_path": configuration_path},
            )
            flows_to_serve.append(flow_deployment)

            # then we deploy the corresponding process flow
            specific_name = flow_name + "_process_flow"
            flow_function = find_flow(specific_name)
            concurrency_value = config["flows"][flow_name].get("concurrency_limit", None)
            concurrency_config = ConcurrencyLimitConfig(
                    limit=concurrency_value,
                    collision_strategy=ConcurrencyLimitStrategy.ENQUEUE,
                ) if concurrency_value else None
            flow_deployment = flow_function.to_deployment(
                name=specific_name,
                description="Process: " + specific_description,
                tags = ["process"] + specific_tags,
                parameters={"pipeline_config_path": configuration_path},
                concurrency_limit=concurrency_config,
            )
            flows_to_serve.append(flow_deployment)

    if include_control:
        # there are special control flows that manage the pipeline instead of processing data
        # time to kick those off!
        for flow_name in config["control"]:
            flow_function = find_flow(flow_name, "control")
            concurrency_config = ConcurrencyLimitConfig(
                    limit=1,
                    collision_strategy=ConcurrencyLimitStrategy.CANCEL_NEW,
                )
            flow_deployment = flow_function.to_deployment(
                name=flow_name,
                description=config["control"][flow_name].get("description", ""),
                tags=["control"],
                cron=config["control"][flow_name].get("schedule", "* * * * *"),
                parameters={"pipeline_config_path": configuration_path},
                concurrency_limit=concurrency_config,
            )
            flows_to_serve.append(flow_deployment)
    return flows_to_serve

def run_data(configuration_path):
    with get_client(sync_client=True) as client:
        client.create_concurrency_limit(tag="reproject", concurrency_limit=50)
        client.create_concurrency_limit(tag="image_loader", concurrency_limit=50)
    configuration_path = str(Path(configuration_path).resolve())
    serve(*construct_flows_to_serve(configuration_path, include_control=False, include_data=True))

def run_control(configuration_path):
    configuration_path = str(Path(configuration_path).resolve())
    serve(*construct_flows_to_serve(configuration_path, include_control=True, include_data=False))

def run(configuration_path, launch_prefect=False, launch_dask_cluster=False):
    now = datetime.now()

    configuration_path = str(Path(configuration_path).resolve())
    output_path = f"punchpipe_{now.strftime('%Y%m%d_%H%M%S')}.txt"

    print()
    print(f"Launching punchpipe at {now} with configuration: {configuration_path}")
    print(f"Terminal logs from punchpipe are in {output_path}")


    with open(output_path, "a") as f:
        shutdown_expected = False
        prefect_process = None
        prefect_services_process = None
        cluster_process = None
        data_process = None
        control_process = None
        try:
            numa_prefix_control = ["numactl", "--localalloc", "--physcpubind=0-11"]
            numa_prefix_workers = ["numactl", "--localalloc", "--physcpubind=12-63,64-125,192-255"]
            if launch_prefect:
                print("Launcing prefect")
                prefect_process = subprocess.Popen(
                    [*numa_prefix_control, "prefect", "server", "start", "--no-services"], stdout=f, stderr=f)
                time.sleep(5)
                # Separating the server and the background services may help avoid overwhelming the database connections
                # https://github.com/PrefectHQ/prefect/issues/16299#issuecomment-2698732783
                prefect_services_process = subprocess.Popen(
                    [*numa_prefix_control, "prefect", "server", "services", "start"], stdout=f, stderr=f)

            if launch_dask_cluster:
                cluster_process = subprocess.Popen([*numa_prefix_workers, "punchpipe_cluster", configuration_path],
                                                   stdout=f, stderr=f)
            monitor_process = subprocess.Popen([*numa_prefix_control, "gunicorn",
                                                "-b", "0.0.0.0:8050",
                                                "--chdir", THIS_DIR + "/monitor",
                                                "app:server"],
                                               stdout=f, stderr=f)
            time.sleep(1)
            Variable.set("punchpipe_config", configuration_path, overwrite=True)

            # These processes send a _lot_ of output, so we let it go to the screen instead of making the log file
            # enormous
            def data_process_launcher() -> subprocess.Popen:
                return subprocess.Popen([*numa_prefix_workers, "punchpipe", "serve-data", configuration_path])

            def control_process_launcher() -> subprocess.Popen:
                return subprocess.Popen([*numa_prefix_control, "punchpipe", "serve-control", configuration_path])

            data_process = data_process_launcher()
            control_process = control_process_launcher()

            if launch_prefect is not None:
                print("Launched Prefect dashboard on http://localhost:4200/")
            print("Launched punchpipe monitor on http://localhost:8050/")
            print("Launched dask cluster on http://localhost:8786/")
            print("Dask dashboard available at http://localhost:8787/")
            print("Use ctrl-c to exit.")

            time.sleep(10)
            while True:
                # `.poll()` updates but does not return the object's returncode attribute
                if cluster_process is not None:
                    cluster_process.poll()
                control_process.poll()
                data_process.poll()
                if launch_prefect:
                    prefect_process.poll()
                    prefect_services_process.poll()
                    if prefect_process.returncode is not None or prefect_services_process.returncode is not None:
                        print("Prefect process exited unexpectedly")
                        break
                if cluster_process is not None and cluster_process.returncode is not None:
                    print("Cluster process exited unexpectedly")
                    break
                # Core processes are still running. Now check worker processes, which we can restart safely
                if control_process.returncode is not None:
                    print(f"Restarted control process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    control_process = control_process_launcher()
                if data_process.returncode is not None:
                    print(f"Restarted data process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    data_process = data_process_launcher()
                time.sleep(10)
            raise RuntimeError
        except KeyboardInterrupt:
            print("Shutting down.")
            shutdown_expected = True
        except Exception as e:
            print(f"Received error: {e}")
            print(traceback.format_exc())
        finally:
            control_process.terminate() if control_process else None
            data_process.terminate() if data_process else None
            control_process.wait() if control_process else None
            data_process.wait() if data_process else None
            time.sleep(1)
            if launch_prefect:
                prefect_services_process.terminate() if prefect_services_process else None
                prefect_services_process.wait() if prefect_services_process else None
                time.sleep(3)
                prefect_process.terminate() if prefect_process else None
                prefect_process.wait() if prefect_process else None
                time.sleep(3)
            cluster_process.terminate() if cluster_process else None
            monitor_process.terminate() if monitor_process else None
            cluster_process.wait() if cluster_process else None
            monitor_process.wait() if monitor_process else None
            print()
            if shutdown_expected:
                print("punchpipe safely shut down")
            else:
                print("punchpipe abruptly shut down")


def clean_replay(input_file: str, configuration_path, write: bool = True, window_in_days: int | None = None, reference_date : None | datetime = None) -> None | pd.DataFrame:
    """Clean replay requests"""
    if reference_date is None:
        reference_date = datetime.now()

    config = load_pipeline_configuration(configuration_path)

    input_path = Path(input_file)

    output_file = input_path.parent / f"merged_{input_path.name}"
    output_file_soc = input_path.parent / f"merged_soc_{input_path.name}"

    df = pd.read_csv(input_file)

    df = df.sort_values("start_block").reset_index(drop=True)

    if window_in_days is None:
        window_in_days = config["replay"]["window_in_days"]

    if window_in_days is not None:
        df = df[pd.to_datetime(df["start_time"]) >= (reference_date - timedelta(days=window_in_days))]
        df = df[pd.to_datetime(df["start_time"]) <= reference_date]

    blocks_science = config["replay"]["science_blocks"]

    df = df[df["start_block"] >= blocks_science[0]]
    df = df[df["start_block"] <= blocks_science[1]]

    merged_blocks = []

    for _, row in df.iterrows():
        start = row["start_block"]
        length = row["replay_length"]
        end = start + length
        start_time = row["start_time"]

        if length < 0:
            length = length + blocks_science[1] - blocks_science[0] + 1
            wrapped_replay = True
        else:
            wrapped_replay = False

        if merged_blocks and (start <= merged_blocks[-1]["end"]):
            last_block = merged_blocks[-1]

            print(f"Overlap found: Block {start}-{end} overlaps with {last_block['start_block']}-{last_block['end']}")

            new_end = max(last_block["end"], end)
            new_length = new_end - last_block["start_block"] + 1

            merged_blocks[-1]["end"] = new_end
            merged_blocks[-1]["replay_length"] = new_length

            print(f"Merged into: Block {last_block['start_block']}-{new_end} (length: {new_length})")

        elif merged_blocks and wrapped_replay and (end >= merged_blocks[0]["start_block"]):
            first_block = merged_blocks[0]

            print(f"Overlap found: Block {start}-{end} overlaps with {first_block['start_block']}-{first_block['end']}")

            new_end = max(first_block["end"], end)
            new_length = new_end - start + blocks_science[1] - blocks_science[0] + 1

            merged_blocks[0]["start_block"] = start
            merged_blocks[0]["end"] = new_end
            merged_blocks[0]["replay_length"] = new_length

            print(f"Merged into: Block {first_block['start_block']}-{new_end} (length: {new_length})")

        else:
            merged_blocks.append({
                "start_time": start_time,
                "start_block": start,
                "replay_length": length,
                "end": end,
            })

    if len(merged_blocks) != 0:
        result_df = pd.DataFrame(merged_blocks).drop("end", axis=1)

        print(f"\nOriginal blocks: {len(df)}")
        print(f"Merged blocks: {len(result_df)}")
        print(f"Blocks merged: {len(df) - len(result_df)}")

        result_df = result_df.sort_values("start_time").reset_index(drop=True)
    else:
        result_df = pd.DataFrame([])

    if write:
        result_df.to_csv(output_file_soc, index=False)
        print(f"Results written to {output_file_soc}")

        with open(output_file.with_suffix(".txt"), "w") as f:
            for _, row in result_df.iterrows():
                f.write(f"start mops_fsw_start_fast_replay(xfi,{row['start_block']},{row['replay_length']})\n")
    else:
        return result_df
