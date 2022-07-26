from typing import Optional
from datetime import datetime
from punchbowl.data import PUNCHData
from prefect import task, get_run_logger


@task
def correct_psf(data_object):
    logger = get_run_logger()
    logger.info("correct_psf started")
    # do PSF correction in here
    logger.info("correct_psf finished")
    data_object.add_history(datetime.now(), "LEVEL1-correct_psf", "PSF corrected")
    return data_object
