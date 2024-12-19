from math import floor
from datetime import datetime

import numpy as np
import remove_starfield
from astropy.wcs import WCS
from dateutil.parser import parse as parse_datetime_str
from ndcube import NDCube
from prefect import flow, get_run_logger
from remove_starfield import ImageHolder, ImageProcessor, Starfield
from remove_starfield.reducers import PercentileReducer

from punchbowl.data import NormalizedMetadata, load_ndcube_from_fits
from punchbowl.data.wcs import calculate_helio_wcs_from_celestial
from punchbowl.prefect import punch_task


class PUNCHImageProcessor(ImageProcessor):
    """Special loader for PUNCH data."""

    def __init__(self, layer: int, apply_mask: bool = True, key: str = " ") -> None:
        """Create PUNCHImageProcessor."""
        self.layer = layer
        self.apply_mask = apply_mask
        self.key = key

    def load_image(self, filename: str) -> ImageHolder:
        """Load an image."""
        cube = load_ndcube_from_fits(filename, key=self.key)
        data = cube.data[self.layer]
        if self.apply_mask:
            data[np.isclose(cube.uncertainty.array[self.layer], 0, atol=1E-30)] = np.nan
        return ImageHolder(data, cube.wcs.celestial, cube.meta)


@flow(log_prints=True, timeout_seconds=21_600)
def generate_starfield_background(
        filenames: list[str],
        map_scale: float = 0.01,
        target_mem_usage: float = 1000,
        n_procs: int | None = None,
        reference_time: datetime | None = None) -> [NDCube, NDCube]:
    """Create a background starfield_bg map from a series of PUNCH images over a long period of time."""
    logger = get_run_logger()

    if reference_time is None:
        reference_time = datetime.now()
    elif isinstance(reference_time, str):
        reference_time = parse_datetime_str(reference_time)

    logger.info("construct_starfield_background started")

    # create an empty array to fill with data
    #   open the first file in the list to ge the shape of the file
    if len(filenames) == 0:
        msg = "filenames cannot be empty"
        raise ValueError(msg)

    shape = [int(floor(132 / map_scale)), int(floor(360 / map_scale))]
    starfield_wcs = WCS(naxis=2)
    # n.b. it seems the RA wrap point is chosen so there's 180 degrees
    # included on either side of crpix
    crpix = [shape[1] / 2 + .5, shape[0] / 2 + .5]
    starfield_wcs.wcs.crpix = crpix
    starfield_wcs.wcs.crval = 270, -23.5
    starfield_wcs.wcs.cdelt = map_scale, map_scale
    starfield_wcs.wcs.ctype = "RA---CAR", "DEC--CAR"
    starfield_wcs.wcs.cunit = "deg", "deg"
    starfield_wcs.array_shape = shape

    logger.info("Starting m starfield")
    starfield_m = remove_starfield.build_starfield_estimate(
        filenames,
        attribution=False,
        frame_count=False,
        reducer=PercentileReducer(10),
        starfield_wcs=starfield_wcs,
        n_procs=n_procs,
        processor=PUNCHImageProcessor(0, apply_mask=True, key="A"),
        target_mem_usage=target_mem_usage)
    logger.info("Ending m starfield")

    logger.info("Starting z starfield")
    starfield_z = remove_starfield.build_starfield_estimate(
        filenames,
        attribution=False,
        frame_count=False,
        reducer=PercentileReducer(10),
        starfield_wcs=starfield_wcs,
        n_procs=n_procs,
        processor=PUNCHImageProcessor(1, apply_mask=True, key="A"),
        target_mem_usage=target_mem_usage)
    logger.info("Ending z starfield")


    logger.info("Starting p starfield")
    starfield_p = remove_starfield.build_starfield_estimate(
        filenames,
        attribution=False,
        frame_count=False,
        reducer=PercentileReducer(10),
        starfield_wcs=starfield_wcs,
        n_procs=n_procs,
        processor=PUNCHImageProcessor(2, apply_mask=True, key="A"),
        target_mem_usage=target_mem_usage)
    logger.info("Ending p starfield")

    # create an output PUNCHdata object
    logger.info("Preparing to create outputs")

    meta = NormalizedMetadata.load_template("PSM", "3")
    meta["DATE-OBS"] = reference_time.isoformat()
    meta["DATE-BEG"] = reference_time.isoformat()
    meta["DATE-END"] = reference_time.isoformat()
    meta["DATE-AVG"] = reference_time.isoformat()
    meta["DATE"] = datetime.now().isoformat()

    out_wcs, _ = calculate_helio_wcs_from_celestial(starfield_m.wcs, meta.astropy_time, starfield_m.starfield.shape)
    output = NDCube(np.stack([starfield_m.starfield, starfield_z.starfield, starfield_p.starfield], axis=0),
                    wcs=out_wcs, meta=meta)
    output.meta.history.add_now("LEVEL3-starfield_background", "constructed starfield_bg model")

    logger.info("construct_starfield_background finished")
    return [output]


@punch_task
def subtract_starfield_background_task(data_object: NDCube,
                                       starfield_background_path: str | None) -> NDCube:
    """
    Subtracts a background starfield from an input data frame.

    checks the dimensions of input data frame and background starfield match and
    subtracts the background starfield from the data frame of interest.

    Parameters
    ----------
    data_object : NDCube
        A NDCube data frame to be background subtracted

    starfield_background_path : str
        path to a NDCube background starfield map

    Returns
    -------
    NDCube
        A background starfield subtracted data frame

    """
    logger = get_run_logger()
    logger.info("subtract_starfield_background started")

    if starfield_background_path is None:
        output = data_object
        output.meta.history.add_now("LEVEL3-subtract_starfield_background",
                                           "starfield subtraction skipped since path is empty")
    else:
        star_datacube = load_ndcube_from_fits(starfield_background_path, key="A")
        # data_wcs = calculate_celestial_wcs_from_helio(data_object.wcs.celestial,
        #                                               data_object.meta.astropy_time,
        #                                               data_object.data.shape[-2:])
        map_scale = 0.01
        shape = [int(floor(132 / map_scale)), int(floor(360 / map_scale))]
        starfield_wcs = WCS(naxis=2)
        # n.b. it seems the RA wrap point is chosen so there's 180 degrees
        # included on either side of crpix
        crpix = [shape[1] / 2 + .5, shape[0] / 2 + .5]
        starfield_wcs.wcs.crpix = crpix
        starfield_wcs.wcs.crval = 270, -23.5
        starfield_wcs.wcs.cdelt = map_scale, map_scale
        starfield_wcs.wcs.ctype = "RA---CAR", "DEC--CAR"
        starfield_wcs.wcs.cunit = "deg", "deg"
        starfield_wcs.array_shape = shape

        starfield_model = Starfield(np.stack((star_datacube.data, star_datacube.uncertainty.array)), starfield_wcs)

        subtracted = starfield_model.subtract_from_image(
            NDCube(data=np.stack((data_object.data, data_object.uncertainty.array)),
                   wcs=data_object.wcs.celestial,
                   meta=data_object.meta),
            processor=PUNCHImageProcessor(0, key="A"))

        data_object.data[...] = subtracted.subtracted[0]
        data_object.uncertainty.array[...] -= subtracted.subtracted[1]
        data_object.meta.history.add_now("LEVEL3-subtract_starfield_background", "subtracted starfield background")
        output = data_object
    logger.info("subtract_starfield_background finished")


    return output


def create_empty_starfield_background(data_object: NDCube) -> np.ndarray:
    """Create an empty starfield background map."""
    return np.zeros_like(data_object.data)
