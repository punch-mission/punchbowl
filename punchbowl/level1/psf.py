from datetime import datetime

from prefect import task, get_run_logger

from punchbowl.data import PUNCHData


@task
def correct_psf_task(data_object: PUNCHData) -> PUNCHData:
    logger = get_run_logger()
    logger.info("correct_psf started")
    # TODO: do PSF correction in here
    logger.info("correct_psf finished")
    data_object.add_history(datetime.now(), "LEVEL1-correct_psf", "PSF corrected")
    return data_object
