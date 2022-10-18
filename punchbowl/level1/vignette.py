from datetime import datetime

from prefect import task, get_run_logger

from punchbowl.data import PUNCHData


@task
def correct_vignetting_task(data_object: PUNCHData) -> PUNCHData:
    logger = get_run_logger()
    logger.info("correct_vignetting started")
    # do vignetting correction in here
    logger.info("correct_vignetting finished")
    data_object.add_history(
        datetime.now(), "LEVEL1-correct_vignetting", "Vignetting corrected"
    )
    return data_object
