"""
This process starts a Dask cluster and then monitors for changes to the cluster size in the configuration file
"""
import os
import time
import argparse
import traceback
from pathlib import Path

from dask.distributed import LocalCluster

from punchbowl.auto.control.util import load_pipeline_configuration


def main():
    """Run a Dask cluster for the pipeline"""
    parser = argparse.ArgumentParser(prog="punchpipe-cluster")
    parser.add_argument("config", type=str, help="Path to config.")
    args = parser.parse_args()

    configuration_path = str(Path(args.config).resolve())
    config = load_pipeline_configuration(configuration_path)
    config_mtime = os.path.getmtime(configuration_path)

    cluster = LocalCluster(n_workers=config["dask_cluster"]["n_workers"],
                           threads_per_worker=config["dask_cluster"]["n_threads_per_worker"],
                           scheduler_port=8786)
    try:
        while True:
            time.sleep(5)
            if not os.path.exists(configuration_path):
                # In case the file is being re-written right now. (This has happened and crashed this process!)
                time.sleep(1)
            if (cur_mtime := os.path.getmtime(configuration_path)) != config_mtime:
                config_mtime = cur_mtime
                config = load_pipeline_configuration(configuration_path)
                # This tells the cluster to add or remove workers
                cluster.scale(config["dask_cluster"]["n_workers"])
    except Exception as e:
        print(f"Received error: {e}")
        print(traceback.format_exc())
        print("Stopping cluster")
        cluster.close()
