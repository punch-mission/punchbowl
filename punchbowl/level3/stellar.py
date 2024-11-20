from datetime import datetime

import numpy as np
import remove_starfield
from ndcube import NDCube
from prefect import flow, get_run_logger
from remove_starfield import ImageHolder, ImageProcessor, Starfield
from remove_starfield.reducers import GaussianReducer

from punchbowl.data import NormalizedMetadata, load_ndcube_from_fits
from punchbowl.data.wcs import calculate_celestial_wcs_from_helio, calculate_helio_wcs_from_celestial
from punchbowl.prefect import punch_task


class PUNCHImageProcessor(ImageProcessor):
    """Special loader for PUNCH data."""

    def __init__(self, layer: int) -> None:
        """Create PUNCHImageProcessor."""
        self.layer = layer

    def load_image(self, filename: str) -> ImageHolder:
        """Load an image."""
        cube = load_ndcube_from_fits(filename, key="A")
        return ImageHolder(cube.data[self.layer], cube.wcs.celestial, cube.meta)


@flow(log_prints=True)
def generate_starfield_background(
        filenames: list[str],
        n_sigma: float = 5,
        map_scale: float = 0.01,
        target_mem_usage: float = 1000) -> [NDCube, NDCube]:
    """Create a background starfield_bg map from a series of PUNCH images over a long period of time."""
    logger = get_run_logger()
    logger.info("construct_starfield_background started")

    # create an empty array to fill with data
    #   open the first file in the list to ge the shape of the file
    if len(filenames) == 0:
        msg = "filenames cannot be empty"
        raise ValueError(msg)

    ra_bounds = (0, 180)
    dec_bounds = (-90, 90)

    logger.info("Starting m starfield")
    starfield_m = remove_starfield.build_starfield_estimate(
        filenames,
        attribution=False,
        frame_count=False,
        reducer=GaussianReducer(n_sigma=n_sigma),
        ra_bounds=ra_bounds,
        dec_bounds=dec_bounds,
        map_scale=map_scale,
        processor=PUNCHImageProcessor(0),
        target_mem_usage=target_mem_usage)
    logger.info("Ending m starfield")


    logger.info("Starting z starfield")
    starfield_z = remove_starfield.build_starfield_estimate(
        filenames,
        attribution=False,
        frame_count=False,
        ra_bounds=ra_bounds,
        dec_bounds=dec_bounds,
        reducer=GaussianReducer(n_sigma=n_sigma),
        map_scale=map_scale,
        processor=PUNCHImageProcessor(1),
        target_mem_usage=target_mem_usage)
    logger.info("Ending z starfield")


    logger.info("Starting p starfield")
    starfield_p = remove_starfield.build_starfield_estimate(
        filenames,
        attribution=False,
        frame_count=False,
        ra_bounds=ra_bounds,
        dec_bounds=dec_bounds,
        reducer=GaussianReducer(n_sigma=n_sigma),
        map_scale=map_scale,
        processor=PUNCHImageProcessor(2),
        target_mem_usage=target_mem_usage)
    logger.info("Ending p starfield")


    # create an output PUNCHdata object
    logger.info("Preparing to create outputs")

    meta = NormalizedMetadata.load_template("PSM", "3")
    meta["DATE-OBS"] = str(datetime(2024, 8, 1, 12, 0, 0,
                                    tzinfo=datetime.timezone.utc))
    out_wcs, _ = calculate_helio_wcs_from_celestial(starfield_m.wcs, meta.astropy_time, starfield_m.starfield.shape)
    output_before = NDCube(np.stack([starfield_m.starfield, starfield_z.starfield, starfield_p.starfield], axis=0),
                    wcs=out_wcs, meta=meta)
    output_before.meta.history.add_now("LEVEL3-starfield_background", "constructed starfield_bg model")

    meta = NormalizedMetadata.load_template("PSM", "3")
    meta["DATE-OBS"] = str(datetime(2024, 12, 1, 12, 0, 0,
                                    tzinfo=datetime.timezone.utc))
    out_wcs, _ = calculate_helio_wcs_from_celestial(starfield_m.wcs, meta.astropy_time, starfield_m.starfield.shape)
    output_after = NDCube(np.stack([starfield_m.starfield, starfield_z.starfield, starfield_p.starfield], axis=0),
                    wcs=out_wcs, meta=meta)
    output_after.meta.history.add_now("LEVEL3-starfield_background", "constructed starfield_bg model")

    logger.info("construct_starfield_background finished")
    return [output_before, output_after]


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
        star_datacube = load_ndcube_from_fits(starfield_background_path)
        data_wcs = calculate_celestial_wcs_from_helio(data_object.wcs.celestial,
                                                      data_object.meta.astropy_time,
                                                      data_object.data.shape[-2:])
        starfield_model = Starfield(star_datacube.data, star_datacube.wcs.celestial)

        starfield_subtracted_data = starfield_model.subtract_from_image(
            NDCube(data=data_object.data,
                   wcs=data_wcs,
                   meta=data_object.meta),
            processor=PUNCHImageProcessor(0))

        starfield_subtracted_uncertainty = starfield_model.subtract_from_image(
            NDCube(data=data_object.uncertainty.array,
                   wcs=data_wcs,
                   meta=data_object.meta),
            processor=PUNCHImageProcessor(0))

        data_object.data[...] = starfield_subtracted_data.subtracted
        data_object.uncertainty.array[...] -= starfield_subtracted_uncertainty.subtracted
        data_object.meta.history.add_now("LEVEL3-subtract_starfield_background", "subtracted starfield background")
        output = data_object
    logger.info("subtract_f_corona_background finished")

    return output


def create_empty_starfield_background(data_object: NDCube) -> np.ndarray:
    """Create an empty starfield background map."""
    return np.zeros_like(data_object.data)
