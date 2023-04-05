# Core Python imports
import pathlib

# Third party imports
import numpy as np
from prefect import get_run_logger, task

# Punchbowl imports
from punchbowl.data import PUNCHData

THIS_DIRECTORY = pathlib.Path(__file__).parent.resolve()


def flag_punchdata(data_object: PUNCHData, bad_pixel_map: np.ndarray = None) -> PUNCHData:
    """
    Core bad pixel flagging function.

    Parameters
    ----------
    data_object
        Input PUNCHData object

    bad_pixel_map
        Specified bad pixel map

    Returns
    -------
    PUNCHData object with bad pixels flagged in the primary data and uncertainty arrays

    """

    # Flag bad data in the associated uncertainty array (coded with infinity)
    data_object.uncertainty.array[bad_pixel_map] = np.inf

    return data_object


@task
def flag_task(data_object: PUNCHData, bad_pixel_map: PUNCHData) -> PUNCHData:
    """
    Pipeline task for bad pixel flagging

    Parameters
    ----------
    data_object
        Input PUNCHData object

    bad_pixel_map
        PUNCHData to store bad pixel map

    Returns
    -------
    PUNCHData object with bad pixels flagged in the primary data and uncertainty arrays

    """

    logger = get_run_logger()
    logger.info("flagging started")

    # Call data flagging function
    data_object = flag_punchdata(data_object, bad_pixel_map.data)

    logger.info("flagging finished")
    data_object.meta.history.add_now("LEVEL1-flagging", "image flagged")
    return data_object
