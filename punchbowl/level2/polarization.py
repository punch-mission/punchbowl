from datetime import datetime

from prefect import task, get_run_logger

from punchbowl.data import PUNCHData


@task
def resolve_polarization_task(data_object: PUNCHData) -> PUNCHData:
    logger = get_run_logger()
    logger.info("resolve_polarization started")
    # TODO: actually do the resolution
    logger.info("resolve_polarization ended")
    data_object.add_history(datetime.now(), "LEVEL2-resolve_polarization", "polarization resovled")
    return data_object
