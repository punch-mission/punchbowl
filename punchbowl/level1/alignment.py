from typing import Optional
from datetime import datetime
from punchbowl.data import PUNCHData
from prefect import task, get_run_logger


@task
def align(data_object):
    logger = get_run_logger()
    logger.info("alignment started")
    # do alignment in here
    logger.info("alignment finished")
    data_object.add_history(datetime.now(), "LEVEL1-Align", "alignment done")
    return data_object