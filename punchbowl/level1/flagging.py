# Core Python imports
import pathlib
import os

# Third party imports
import numpy as np
from prefect import task, get_run_logger
from astropy.io import fits

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
def flag_task(data_object: PUNCHData, bad_pixel_filename: str) -> PUNCHData:
    """
    Pipeline task for bad pixel flagging

    Parameters
    ----------
    data_object
        Input PUNCHData object

    bad_pixel_filename
        Path to bad pixel calibration file

    Returns
    -------
    PUNCHData object with bad pixels flagged in the primary data and uncertainty arrays

    """

    logger = get_run_logger()
    logger.info("flagging started")

    # Read bad pixel map from file
    if bad_pixel_filename is None:
        raise Exception('Must provide bad pixel map path')

    bad_pixel_map = PUNCHData.from_fits(bad_pixel_filename)

    # Call data flagging function
    data_object = flag_punchdata(data_object, bad_pixel_map.data.astype(int))

    logger.info("flagging finished")
    data_object.meta.history.add_now("LEVEL1-flagging", "image flagged")
    return data_object
