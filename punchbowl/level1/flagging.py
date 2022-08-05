from typing import Optional
from datetime import datetime
from punchbowl.data import PUNCHData
from prefect import task, get_run_logger


@task
def flag(data_object):
    logger = get_run_logger()
    logger.info("flagging started")
    # do despiking in here
    logger.info("flagging finished")
    data_object.add_history(datetime.now(), "LEVEL1-flagging", "image flagged")
    return data_object
