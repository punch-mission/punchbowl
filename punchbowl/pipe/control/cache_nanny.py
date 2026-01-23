import os
import sys
import time

from prefect import flow, get_run_logger
from prefect.variables import Variable

from punchbowl.pipe.control.cache_layer import manager
from punchbowl.pipe.control.util import load_pipeline_configuration


@flow
def cache_nanny(pipeline_config_path: str):
    logger = get_run_logger()

    pipeline_config = load_pipeline_configuration(pipeline_config_path)
    new_state = pipeline_config["cache_layer"]["cache_enabled"] and (sys.version_info.minor >= 13)
    old_state = Variable.get("use_shm_cache", "unset")
    logger.info(f"'cache_enabled' is {old_state}")
    if new_state != old_state:
        Variable.set("use_shm_cache", new_state, overwrite=True)
        logger.info(f"Changed 'cache_enabled' from {old_state} to {new_state}")

    cache_files = manager.get_existing_cache_files()

    cache_files = [
        [
            (stat := os.stat(file)).st_atime,
            stat.st_size,
            file,
        ] for file in cache_files if os.path.isfile(file)
    ]
    cache_files.sort(key=lambda x: x[0])

    n_removed = 0
    size_removed = 0
    cutoff_time = time.time() - pipeline_config["cache_layer"]["max_age_hours"] * 3600
    while cache_files and cache_files[0][0] < cutoff_time:
        _, size, path = cache_files.pop(0)
        os.remove(path)
        n_removed += 1
        size_removed += size
    logger.info(f"Removed {n_removed} cache entries ({size_removed/1e6:.1f} MB) for age")

    total_size = sum(f[1] for f in cache_files)
    excess = total_size - pipeline_config["cache_layer"]["max_size_MB"] * 1e6
    n_removed = 0
    size_removed = 0
    while excess > 0:
        _, size, path = cache_files.pop(0)
        os.remove(path)
        excess -= size
        n_removed += 1
        size_removed += size
    logger.info(f"Removed {n_removed} cache entries ({size_removed/1e6:.1f} MB) for size")
