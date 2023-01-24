from datetime import datetime

from prefect import task, get_run_logger

from punchbowl.data import PUNCHData


@task
def flag_task(data_object: PUNCHData) -> PUNCHData:
    logger = get_run_logger()
    logger.info("flagging started")
    # TODO: do flagging in here
    logger.info("flagging finished")
    data_object.meta.history.add_now("LEVEL1-flagging", "image flagged")
    return data_object
