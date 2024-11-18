from datetime import datetime, timedelta

import numpy as np
import remove_starfield
from ndcube import NDCube
from prefect import flow, get_run_logger
from remove_starfield import ImageHolder, ImageProcessor, Starfield
from remove_starfield.reducers import GaussianReducer

from build.lib.punchbowl.level3.f_corona_model import subtract_f_corona_background
from punchbowl.data import NormalizedMetadata, load_ndcube_from_fits
from punchbowl.prefect import punch_task


class PUNCHImageProcessor(ImageProcessor):
    def __init__(self, layer, before_f_corona_path, after_f_corona_path):
        self.layer = layer
        self.before_f_corona = load_ndcube_from_fits(before_f_corona_path)
        self.after_f_corona = load_ndcube_from_fits(after_f_corona_path)

    def load_image(self, filename: str) -> ImageHolder:
        cube = load_ndcube_from_fits(filename)
        subtracted = subtract_f_corona_background(cube, self.before_f_corona, self.after_f_corona)
        return ImageHolder(subtracted.data[self.layer], cube.wcs[self.layer], cube.meta)

@flow
def generate_starfield_background(
        filenames: list[str],
        before_f_corona: str,
        after_f_corona: str,
        n_sigma: float = 5,
        map_scale: float = 0.01,
        target_mem_usage: float = 1000) -> NDCube:
    """Create a background starfield_bg map from a series of PUNCH images over a long period of time."""
    logger = get_run_logger()
    logger.info("construct_starfield_background started")

    # create an empty array to fill with data
    #   open the first file in the list to ge the shape of the file
    if len(filenames) == 0:
        msg = "filenames cannot be empty"
        raise ValueError(msg)

    starfield_m = remove_starfield.build_starfield_estimate(
        filenames,
        attribution=True,
        frame_count=True,
        reducer=GaussianReducer(n_sigma=n_sigma),
        map_scale=map_scale,
        processor=PUNCHImageProcessor(0, before_f_corona, after_f_corona),
        target_mem_usage=target_mem_usage)

    starfield_z = remove_starfield.build_starfield_estimate(
        filenames,
        attribution=True,
        frame_count=True,
        reducer=GaussianReducer(n_sigma=n_sigma),
        map_scale=map_scale,
        processor=PUNCHImageProcessor(1, before_f_corona, after_f_corona),
        target_mem_usage=target_mem_usage)

    starfield_p = remove_starfield.build_starfield_estimate(
        filenames,
        attribution=True,
        frame_count=True,
        reducer=GaussianReducer(n_sigma=n_sigma),
        map_scale=map_scale,
        processor=PUNCHImageProcessor(2, before_f_corona, after_f_corona),
        target_mem_usage=target_mem_usage)

    # create an output PUNCHdata object
    meta = NormalizedMetadata.load_template("PSM", "3")
    meta["DATE-OBS"] = str(datetime.now()-timedelta(days=60))
    output = NDCube(np.stack([starfield_m, starfield_z, starfield_p], axis=0),
                    wcs=starfield_m.wcs, meta=meta)

    logger.info("construct_starfield_background finished")
    output.meta.history.add_now("LEVEL3-starfield_background", "constructed starfield_bg model")

    return output

def subtract_starfield_background(data_object: NDCube, starfield_background_model: Starfield) -> NDCube:
    """Subtract starfield background."""
    starfield_subtracted_data = starfield_background_model.subtract_from_image(
                                                              data_object,
                                                              processor=remove_starfield.ImageProcessor())

    starfield_subtracted_uncertainty = starfield_background_model.subtract_from_image(
                                                              NDCube(data=data_object.uncertainty.array,
                                                                     wcs=data_object.wcs,
                                                                     meta=data_object.meta),
                                                              processor=remove_starfield.ImageProcessor())

    data_object.data[...] = starfield_subtracted_data.subtracted
    data_object.uncertainty.array[...] = starfield_subtracted_uncertainty.subtracted

    return data_object


@punch_task
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
