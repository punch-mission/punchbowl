
import numpy as np
import remove_starfield
from ndcube import NDCube
from prefect import get_run_logger, task
from remove_starfield import Starfield
from remove_starfield.reducers import GaussianReducer

from punchbowl.data import NormalizedMetadata, load_ndcube_from_fits


def generate_starfield_background(
        data_object: NDCube,
        n_sigma: float = 5,
        map_scale: float = 0.01,
        target_mem_usage: float = 1000) -> NDCube:
    """Create a background starfield_bg map from a series of PUNCH images over a long period of time."""
    logger = get_run_logger()
    logger.info("construct_starfield_background started")

    # create an empty array to fill with data
    #   open the first file in the list to ge the shape of the file
    if len(data_object.data) == 0:
        msg = "data_list cannot be empty"
        raise ValueError(msg)

    starfield_bg = remove_starfield.build_starfield_estimate(
        data_object,
        attribution=True,
        frame_count=True,
        reducer=GaussianReducer(n_sigma=n_sigma),
        map_scale=map_scale,
        processor=remove_starfield.ImageProcessor(),
        target_mem_usage=target_mem_usage)

    # create an output PUNCHdata object
    meta_norm = NormalizedMetadata.from_fits_header(starfield_bg["meta"])
    output = NDCube(starfield_bg.starfield, wcs=starfield_bg.wcs, meta=meta_norm)

    logger.info("construct_starfield_background finished")
    output.meta.history.add_now("LEVEL3-starfield_background", "constructed starfield_bg model")

    return output


def subtract_starfield_background(data_object: NDCube, starfield_background_model: Starfield) -> NDCube:
    """Subtract starfield background."""
    starfield_subtracted_data = Starfield.subtract_from_image(starfield_background_model,
                                                              data_object,
                                                              processor=remove_starfield.ImageProcessor())

    starfield_subtracted_uncertainty = Starfield.subtract_from_image(starfield_background_model,
                                                              NDCube(data=data_object.uncertainty.array,
                                                                     wcs=data_object.wcs,
                                                                     meta=data_object.meta),
                                                              processor=remove_starfield.ImageProcessor())

    data_object.data[...] = starfield_subtracted_data.subtracted
    data_object.uncertainty.array[...] = starfield_subtracted_uncertainty.subtracted

    return data_object


@task
def subtract_starfield_background_task(data_object: NDCube,
                                       starfield_background_path: str | None) -> NDCube:
    """
    Subtracts a background starfield from an input data frame.

    checks the dimensions of input data frame and background starfield match and
    subtracts the background starfield from the data frame of interest.

    Parameters
    ----------
    data_object : punchbowl.data.PUNCHData
        A PUNCHData data frame to be background subtracted

    starfield_background_path : str
        path to a PUNCHData background starfield map

    Returns
    -------
    'punchbowl.data.PUNCHData'
        A background starfield subtracted data frame

    """
    logger = get_run_logger()
    logger.info("subtract_starfield_background started")

    if starfield_background_path is None:
        output = data_object
        output.meta.history.add_now("LEVEL3-fcorona-subtraction",
                                           "F corona subtraction skipped since path is empty")
    else:
        star_data_array = load_ndcube_from_fits(starfield_background_path).data
        output = subtract_starfield_background(data_object, star_data_array)
        output.meta.history.add_now("LEVEL3-subtract_starfield_background", "subtracted starfield background")

    logger.info("subtract_f_corona_background finished")

    return output


def create_empty_starfield_background(data_object: NDCube) -> np.ndarray:
    """Create an empty starfield background map."""
    return np.zeros_like(data_object.data)
