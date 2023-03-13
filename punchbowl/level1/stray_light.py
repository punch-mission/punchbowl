from prefect import get_run_logger, task

from punchbowl.data import PUNCHData


@task
def remove_stray_light_task(data_object: PUNCHData) -> PUNCHData:
    """Prefect task to remove stray light from an image

    Parameters
    ----------
    data_object : PUNCHData
        data to operate on

    Returns
    -------
    PUNCHData
        modified version of the input with the stray light removed
    """
    logger = get_run_logger()
    logger.info("remove_stray_light started")
    # TODO: do stray light removal in here
    logger.info("remove_stray_light finished")
    data_object.meta.history.add_now("LEVEL1-remove_stray_light", "stray light removed")
    return data_object
