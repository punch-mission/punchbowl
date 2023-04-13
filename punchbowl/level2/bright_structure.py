from typing import List

from prefect import get_run_logger, task

from punchbowl.data import PUNCHData


@task
def identify_bright_structures_task(data_list: List[PUNCHData]) -> List[PUNCHData]:
    """Prefect task to perform bright structure identification

    Parameters
    ----------
    data_object : PUNCHData
        data to operate on

    Returns
    -------
    PUNCHData
        modified version of the input data with the bright structures identified
    """
    logger = get_run_logger()
    logger.info("identify_bright_structures_task started")
    # TODO: actually do the identification
    logger.info("identify_bright_structures_task ended")
    for data_object in data_list:
        data_object.meta.history.add_now("LEVEL2-bright_structures",
                                         "bright structure identification completed")
    return data_list
