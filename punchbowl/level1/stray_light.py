from datetime import datetime

from prefect import task, get_run_logger

from punchbowl.data import PUNCHData


@task
def remove_stray_light_task(data_object: PUNCHData) -> PUNCHData:
    logger = get_run_logger()
    logger.info("remove_stray_light started")
    # TODO: do stray light removal in here
    logger.info("remove_stray_light finished")
    data_object.add_history(datetime.now(), "LEVEL1-remove_stray_light", "stray light removed")
    return data_object
