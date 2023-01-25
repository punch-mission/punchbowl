from datetime import datetime

from prefect import task, get_run_logger

from punchbowl.data import PUNCHData


@task
def quality_flag_task(data_object: PUNCHData) -> PUNCHData:
    logger = get_run_logger()
    logger.info("quality_flag started")
    # TODO: actually do the quality flagging
    logger.info("quality_flag ended")
    data_object.meta.history.add_now("LEVEL2-quality_flag", "quality flagging completed")
    return data_object
