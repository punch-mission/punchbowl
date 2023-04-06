from typing import List

from prefect import get_run_logger, task

from punchbowl.data import PUNCHData


@task
def resolve_polarization_task(data_list: List[PUNCHData]) -> List[PUNCHData]:
    """Prefect task to resolve the polarization

    Parameters
    ----------
    data_object : PUNCHData
        data to operate on

    Returns
    -------
    PUNCHData
        modified version of the input with polarization resolved
    """
    logger = get_run_logger()
    logger.info("resolve_polarization started")
    # TODO: actually do the resolution
    logger.info("resolve_polarization ended")
    for data_object in data_list:
        data_object.meta.history.add_now("LEVEL2-resolve_polarization", "polarization resovled")
    return data_list
