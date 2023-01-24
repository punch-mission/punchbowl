# Core Python imports
from datetime import datetime

# Third party imports
import numpy as np
from prefect import task, get_run_logger
from astropy.io import fits

# Punchbowl imports
from punchbowl.data import PUNCHData


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

    # Flag bad data in the primary data array
    data_object.data[bad_pixel_map] = 0

    # Flag bad data in the associated uncertainty array (coded with infinity)
    data_object.uncertainty.array[bad_pixel_map] = np.inf

    return data_object


@task
def flag_task(data_object: PUNCHData, bad_pixel_filename: str = None) -> PUNCHData:
    """
    Pipeline task for bad pixel flagging

    Parameters
    ----------
    data_object
        Input PUNCHData object

    Returns
    -------
    PUNCHData object with bad pixels flagged in the primary data and uncertainty arrays

    """

    logger = get_run_logger()
    logger.info("flagging started")

    # Read bad pixel map from file
    if bad_pixel_filename == None:
        id_str = data_object.id
        bad_pixel_filename = 'data/' + id_str[0:9] + 'DP' + id_str[11:] + '.fits'

    with fits.open(bad_pixel_filename) as hdul:
        bad_pixel_map = hdul[0].data

    # Call data flagging function
    data_object = flag_punchdata(data_object, bad_pixel_map)

    logger.info("flagging finished")
    data_object.add_history(datetime.now(), "LEVEL1-flagging", "image flagged")
    return data_object
