from datetime import datetime

from prefect import task, get_run_logger

from punchbowl.data import PUNCHData


@task
def align_task(data_object: PUNCHData) -> PUNCHData:
    logger = get_run_logger()
    logger.info("alignment started")
    # TODO: do alignment in here
    logger.info("alignment finished")
    data_object.meta.history.add_now("LEVEL1-Align", "alignment done")
    return data_object
