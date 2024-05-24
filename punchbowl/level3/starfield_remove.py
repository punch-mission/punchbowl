import remove_starfield
from remove_starfield.reducers import GaussianReducer
from remove_starfield import Starfield

from typing import Optional

import numpy as np
from prefect import get_run_logger, task

from punchbowl.data import PUNCHData, NormalizedMetadata

def generate_starfield_background(
        data_object: PUNCHData,
        n_sigma: float = 5,
        map_scale: float = 0.01,
        target_mem_usage: float = 1000) -> PUNCHData:
    """Creates a background starfield_bg map from a series of PUNCH images over
    a long period of time.

    Creates a background starfield_bg map


    Parameters
    ----------
    data_object
    n_sigma
    target_mem_usage
    map_scale


    Returns
    -------
    return output_PUNCHobject : ['punchbowl.data.PUNCHData']
        returns an array of the same dimensions as the x and y dimensions of
        the input array
    """
    logger = get_run_logger()
    logger.info("construct_starfield_background started")

    # create an empty array to fill with data
    #   open the first file in the list to ge the shape of the file
    if len(data_object.data) == 0:
        raise ValueError("data_list cannot be empty")

    starfield_bg = remove_starfield.build_starfield_estimate(
        data_object, attribution=True, frame_count=True,
        reducer=GaussianReducer(n_sigma=n_sigma), map_scale=map_scale,
        processor=remove_starfield.ImageProcessor(),
        target_mem_usage=target_mem_usage)  # dec_bounds=(0, 35), ra_bounds=(100, 160), # wcs_key='A'),

    # create an output PUNCHdata object
    meta_norm = NormalizedMetadata.from_fits_header(starfield_bg['meta'])
    output = PUNCHData(starfield_bg.starfield, wcs=starfield_bg.wcs, meta=meta_norm)

    logger.info("construct_starfield_background finished")
    output.meta.history.add_now("LEVEL3-starfield_background", "constructed starfield_bg model")

    return output


def subtract_starfield_background(data_object: PUNCHData, starfield_background_model: Starfield) -> PUNCHData:

    starfield_subtracted_data = Starfield.subtract_from_image(starfield_background_model, data_object,
                                                              processor=remove_starfield.ImageProcessor())

    return data_object.duplicate_with_updates(data=starfield_subtracted_data.subtracted)


@task
def subtract_starfield_background_task(data_object: PUNCHData,
                                       starfield_background_path: Optional[str]) -> PUNCHData:
    """subtracts a background starfield from an input data frame.

    checks the dimensions of input data frame and background starfield match and
    subtracts the background starfield from the data frame of interest.

    Parameters
    ----------
    data_object : punchbowl.data.PUNCHData
        A PUNCHobject data frame to be background subtracted

    starfield_background_path : str
        path to a PUNCHobject background starfield map

    Returns
    -------

    star_subtracted_data : ['punchbowl.data.PUNCHData']
        A background starfield subtracted data frame
    """

    logger = get_run_logger()
    logger.info("subtract_starfield_background started")

    if starfield_background_path is None:
        star_data_array = create_empty_starfield_background(data_object)
    else:
        star_data_array = PUNCHData.from_fits(starfield_background_path).data

    output = subtract_starfield_background(data_object, star_data_array)

    logger.info("subtract_f_corona_background finished")
    output.meta.history.add_now("LEVEL3-subtract_starfield_background", "subtracted starfield background")

    return output


def create_empty_starfield_background(data_object: PUNCHData) -> np.ndarray:
    return np.zeros_like(data_object.data)
