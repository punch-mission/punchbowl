from prefect import task, get_run_logger

from punchbowl.data import PUNCHData


@task
def remove_deficient_pixels_task(data_object: PUNCHData) -> PUNCHData:
    """Prefect task to remove deficient pixels

    Parameters
    ----------
    data_object : PUNCHData
        data to operate on

    Returns
    -------
    PUNCHData
        modified version of the input with the deficient pixels flagged
    """
    logger = get_run_logger()
    logger.info("remove_deficient_pixels started")
    # TODO: do deficient pixel removal in here
    logger.info("remove_deficient_pixels finished")
    data_object.meta.history.add_now("LEVEL1-remove_deficient_pixels", "deficient pixels removed")
    return data_object
