from typing import Optional
from datetime import datetime
from punchbowl.data import PUNCHData
from prefect import task, get_run_logger


@task
def despike(data_object):
    logger = get_run_logger()
    logger.info("despike started")
    # do despiking in here
    logger.info("despike finished")
    data_object.add_history(datetime.now(), "LEVEL1-despike", "image despiked")
    return data_object

