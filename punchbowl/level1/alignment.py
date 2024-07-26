from ndcube import NDCube
from prefect import get_run_logger, task


@task
def align_task(data_object: NDCube) -> NDCube:
    """
    Determine the pointing of the image and updates the metadata appropriately.

    Parameters
    ----------
    data_object : PUNCHData
        data object to align

    Returns
    -------
    PUNCHData
        a modified version of the input with the WCS more accurately determined

    """
    logger = get_run_logger()
    logger.info("alignment started")
    # TODO: do alignment in here
    logger.info("alignment finished")
    data_object.meta.history.add_now("LEVEL1-Align", "alignment done")
    return data_object
