from typing import Optional
from datetime import datetime
from punchbowl.data import PUNCHData
from prefect import task, get_run_logger


@task
def remove_stray_light(data_object):
    logger = get_run_logger()
    logger.info("remove_stray_light started")
    # do stray light removal in here
    logger.info("remove_stray_light finished")
    data_object.add_history(datetime.now(), "LEVEL1-remove_stray_light", "stray light removed")
    return data_object
