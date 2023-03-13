from prefect import get_run_logger, task

from punchbowl.data import PUNCHData


@task
def despike_task(data_object: PUNCHData) -> PUNCHData:
    """Prefect task to perform despiking

    Parameters
    ----------
    data_object : PUNCHData
        data to operate on

    Returns
    -------
    PUNCHData
        a modified version of the input with spikes removed
    """
    logger = get_run_logger()
    logger.info("despike started")
    # TODO: do despiking in here
    logger.info("despike finished")
    data_object.meta.history.add_now("LEVEL1-despike", "image despiked")
    return data_object
