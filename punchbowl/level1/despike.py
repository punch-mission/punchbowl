from datetime import datetime

from prefect import task, get_run_logger

from punchbowl.data import PUNCHData


@task
def despike_task(data_object: PUNCHData) -> PUNCHData:
    logger = get_run_logger()
    logger.info("despike started")
    # TODO: do despiking in here
    logger.info("despike finished")
    data_object.add_history(datetime.now(), "LEVEL1-despike", "image despiked")
    return data_object

