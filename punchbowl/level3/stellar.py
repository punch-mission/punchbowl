import warnings
from math import floor
from datetime import UTC, datetime

import astropy.units as u
import numpy as np
import remove_starfield
from astropy.io import fits
from astropy.io.fits import getheader
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from dateutil.parser import parse as parse_datetime_str
from ndcube import NDCollection, NDCube
from prefect import get_run_logger
from remove_starfield import ImageHolder, ImageProcessor, Starfield
from remove_starfield.reducers import GaussianReducer
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs
from scipy.ndimage import percentile_filter
from scipy.stats import circmean
from solpolpy import resolve
from solpolpy.util import solnorth_from_wcs

from punchbowl.data import NormalizedMetadata, load_ndcube_from_fits, write_ndcube_to_fits
from punchbowl.data.wcs import (
    calculate_celestial_wcs_from_helio,
    calculate_helio_wcs_from_celestial,
    celestial_north_from_wcs,
)
from punchbowl.exceptions import InvalidDataError
from punchbowl.prefect import punch_flow, punch_task
from punchbowl.util import average_datetime, interpolate_data

warnings.filterwarnings("ignore")


def polarize_solar_to_celestial(input_data: NDCube, dtype: None | type = None) -> NDCube:
    """
    Convert polarization from mzpsolar to Celestial frame.

    All images need their polarization converted to Celestial frame
    to generate the background starfield model.
    """
    # Create a data collection for M, Z, P components
    mzp_angles = [-60, 0, 60]*u.degree

    ncols, nrows = input_data.data[0].shape
    wcs1 = (calculate_helio_wcs_from_celestial(input_data.wcs, input_data.meta.astropy_time, input_data.data.shape)
            .deepcopy().dropaxis(2))
    wcs2 = input_data.wcs.deepcopy().dropaxis(2)

    # Converting polarization w.r.t. Celestial North
    angle_solar_north = solnorth_from_wcs(wcs1, (nrows, ncols))
    angle_celest_north = celestial_north_from_wcs(wcs2, (nrows, ncols))

    zoff = (angle_celest_north.value - angle_solar_north.value) * u.degree
    new_angles = np.stack([zoff - 60 * u.deg, zoff, zoff + 60 * u.deg])

    collection_contents = [
        (label,
         NDCube(data=input_data[i].data,
                wcs=wcs1,
                meta={"POLAR": angle}))
        for label, i, angle in zip(["M", "Z", "P"], [0, 1, 2], mzp_angles, strict=False)
    ]
    data_collection = NDCollection(collection_contents, aligned_axes="all")

    # Resolve data to celestial frame
    celestial_data_collection = resolve(data_collection, "npol", out_angles=new_angles)

    valid_keys = [key for key in celestial_data_collection if key != "alpha"]
    new_data = np.array([celestial_data_collection[key].data for key in valid_keys], dtype=dtype)
    new_wcs = input_data.wcs.copy()

    output_meta = NormalizedMetadata.load_template("PTM", "3")
    output_meta["DATE-OBS"] = input_data.meta["DATE-OBS"].value

    output = NDCube(data=new_data, wcs=new_wcs, meta=output_meta)
    output.meta.history.add_now("LEVEL3-convert2celestial", "Convert mzpsolar to Celestial")

    return output


def polarize_celestial_to_solar(input_data: NDCube, dtype: None | type = None) -> NDCube:
    """
    Convert polarization from Celestial frame to mzpsolar.

    All images need their polarization converted back to Solar frame
    after removing the stellar polarization.
    """
    # Compute new angles for celestial frame
    ncols, nrows = input_data.data[0].shape
    wcs1 = (calculate_helio_wcs_from_celestial(input_data.wcs, input_data.meta.astropy_time, input_data.data.shape)
            .deepcopy().dropaxis(2))
    wcs2 = input_data.wcs.deepcopy().dropaxis(2)

    # Converting polarization w.r.t. Celestial North
    angle_solar_north = solnorth_from_wcs(wcs1, (nrows, ncols))
    angle_celest_north = celestial_north_from_wcs(wcs2, (nrows, ncols))

    zoff = (angle_celest_north.value - angle_solar_north.value) * u.degree
    new_angles = np.stack([zoff - 60 * u.deg, zoff, zoff + 60 * u.deg])

    collection_contents = [
        (f"{np.round(new_angles[i, nrows//2, ncols//2].value)} deg",
         NDCube(data=input_data[i].data,
                wcs=wcs1,
                meta={"POLAR": angle}))
        for i, angle in enumerate(new_angles)
    ]
    data_collection = NDCollection(collection_contents, aligned_axes="all")

    # Resolve data to mzpsolar frame
    solar_data_collection = resolve(data_collection, "mzpsolar", in_angles=new_angles)

    valid_keys = [key for key in solar_data_collection if key != "alpha"]
    new_data = np.array([solar_data_collection[key].data for key in valid_keys], dtype=dtype)
    new_wcs = input_data.wcs.copy()

    output_meta = NormalizedMetadata.load_template("PTM", "3")
    output_meta["DATE-OBS"] = input_data.meta["DATE-OBS"].value

    output = NDCube(data=new_data, wcs=new_wcs, meta=output_meta, uncertainty=input_data.uncertainty)
    output.meta.history.add_now("LEVEL3-convert2mzpsolar", "Convert Celestial to mzpsolar")

    return output


class PUNCHImageProcessor(ImageProcessor):
    """Special loader for PUNCH data."""

    def __init__(self, layer: int | None = None, apply_mask: bool = True, key: str = " ") -> None:
        """Create PUNCHImageProcessor."""
        self.layer: int | None = layer
        self.apply_mask = apply_mask
        self.key = key

    def load_image(self, filename: str) -> ImageHolder:
        """Load an image."""
        cube = load_ndcube_from_fits(filename, key=self.key, include_provenance=False, include_uncertainty=False,
                                     dtype=np.float32)

        if self.apply_mask:
            mask = (cube.data[self.layer] == 0) if self.layer is not None else (cube.data == 0)

        if self.layer is None:  # clear data
            data = cube.data
        else:  # it's polarized
            cube = polarize_solar_to_celestial(cube, dtype=np.float32)
            data = cube.data[self.layer]

        if self.apply_mask:
            data[mask] = np.nan
        return ImageHolder(data, cube.wcs.celestial, cube.meta)


def determine_wcs(filenames: list, map_scale: float) -> WCS:
    """Calculate a tightly-cropped model WCS."""
    # Load a sample of WCSes and see where they fall in the sky
    wcs_sample = []
    filenames = sorted(filenames)
    # Load a sample evenly-spaced through the files, being sure to include the first and last images
    indices = np.linspace(0, len(filenames) - 1, 150, dtype=int)
    for i in indices:
        path = filenames[i]
        with fits.open(path) as hdul:
            wcs = WCS(hdul[1].header, hdul, key="A")
            if hdul[1].header["NAXIS"] == 3:
                    wcs = wcs.dropaxis(2)
            wcs_sample.append(wcs)

    # Get the coordinates of the edge of each image
    ras = []
    decs = []
    xs = np.linspace(-1, wcs_sample[0].array_shape[1], 500)
    ys = np.linspace(-1, wcs_sample[0].array_shape[1], 500)
    edgex = np.concatenate((xs,  # bottom edge
                            np.full(len(ys), xs[-1]),  # right edge
                            xs,  # top edge
                            np.full(len(ys), xs[0])))  # left edge
    edgey = np.concatenate((np.full(len(xs), ys[0]),  # bottom edge
                            ys,  # right edge
                            np.full(len(xs), ys[-1]),  # top edge
                            ys))  # left edge
    for wcs in wcs_sample:
        w = wcs.pixel_to_world(edgex, edgey)
        ras.extend(w.ra.deg.ravel())
        decs.extend(w.dec.deg.ravel())

    # Find the center of all the images
    crval = circmean(ras, low=0, high=360), circmean(decs, low=-90, high=90)

    # Start with an all-sky WCS, which we'll crop in
    shape = [floor(180 / map_scale), floor(360 / map_scale)]
    starfield_wcs = WCS(naxis=2)
    # n.b. it seems the RA wrap point is chosen so there's 180 degrees
    # included on either side of crpix
    starfield_wcs.wcs.crpix = [shape[1] / 2 + .5, shape[0] / 2 + .5]
    starfield_wcs.wcs.crval = crval
    starfield_wcs.wcs.cdelt = map_scale, map_scale
    starfield_wcs.wcs.ctype = "RA---CAR", "DEC--CAR"
    starfield_wcs.wcs.cunit = "deg", "deg"
    starfield_wcs.array_shape = shape

    # Find the crop bounds
    xs, ys = starfield_wcs.world_to_pixel_values(ras, decs)
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    margin = 5 # In degrees
    xmin = int(xmin - margin / map_scale)
    ymin = int(ymin - margin / map_scale)
    xmax = int(xmax + margin / map_scale)
    ymax = int(ymax + margin / map_scale)
    return starfield_wcs[ymin:ymax, xmin:xmax]


@punch_flow(log_prints=True, timeout_seconds=21_600)
def generate_starfield_background(
        filenames: list[str],
        map_scale: float = 0.01,
        target_mem_usage: float = 1000,
        n_procs: int | None = None,
        reference_time: datetime | None = None,
        is_polarized: bool = False,
        out_file: str | None = None) -> NDCube | None :
    """Create a background starfield map from a series of PUNCH images over a long period of time."""
    logger = get_run_logger()

    if reference_time is None:
        reference_time = datetime.now(UTC)
    elif isinstance(reference_time, str):
        reference_time = parse_datetime_str(reference_time)

    logger.info("construct_starfield_background started")

    # create an empty array to fill with data
    # open the first file in the list to ge the shape of the file
    if len(filenames) == 0:
        msg = "filenames cannot be empty"
        raise ValueError(msg)

    starfield_wcs = determine_wcs(filenames, map_scale)

    date_obses = [getheader(f, 1)["DATE-OBS"] for f in filenames]
    times = [datetime.fromisoformat(d) for d in date_obses]

    meta = NormalizedMetadata.load_template("PSM" if is_polarized else "CSM", "3")
    meta["DATE-OBS"] = reference_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    meta["DATE-BEG"] = min(times).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    meta["DATE-END"] = max(times).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    meta["DATE-AVG"] = average_datetime(times).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    if is_polarized:
        logger.info("Starting m starfield")
        starfield_m = remove_starfield.build_starfield_estimate(
            filenames,
            attribution=False,
            frame_count=False,
            reducer=GaussianReducer(),
            starfield_wcs=starfield_wcs,
            n_procs=n_procs,
            processor=PUNCHImageProcessor(0, apply_mask=True, key="A"),
            handle_wrap_point=False,
            dtype=np.float32,
            target_mem_usage=target_mem_usage)
        logger.info("Ending m starfield")
        out_data_m = starfield_m.starfield - percentile_filter(starfield_m.starfield, 5, 10)
        out_data_m[out_data_m < 0] = 0

        logger.info("Starting z starfield")
        starfield_z = remove_starfield.build_starfield_estimate(
            filenames,
            attribution=False,
            frame_count=False,
            reducer=GaussianReducer(),
            starfield_wcs=starfield_wcs,
            n_procs=n_procs,
            processor=PUNCHImageProcessor(1, apply_mask=True, key="A"),
            handle_wrap_point=False,
            dtype=np.float32,
            target_mem_usage=target_mem_usage)
        logger.info("Ending z starfield")
        out_data_z = starfield_z.starfield - percentile_filter(starfield_z.starfield, 5, 10)
        out_data_z[out_data_z < 0] = 0

        logger.info("Starting p starfield")
        starfield_p = remove_starfield.build_starfield_estimate(
            filenames,
            attribution=False,
            frame_count=False,
            reducer=GaussianReducer(),
            starfield_wcs=starfield_wcs,
            n_procs=n_procs,
            processor=PUNCHImageProcessor(2, apply_mask=True, key="A"),
            handle_wrap_point=False,
            dtype=np.float32,
            target_mem_usage=target_mem_usage)
        logger.info("Ending p starfield")
        out_data_p = starfield_p.starfield - percentile_filter(starfield_p.starfield, 5, 10)
        out_data_p[out_data_p < 0] = 0

        out_data = np.stack([out_data_m, out_data_z, out_data_p], axis=0)
        out_wcs = calculate_helio_wcs_from_celestial(starfield_m.wcs, meta.astropy_time, starfield_m.starfield.shape)
    else:
        logger.info("Starting clear starfield")
        starfield_clear = remove_starfield.build_starfield_estimate(
            filenames,
            attribution=False,
            frame_count=False,
            reducer=GaussianReducer(),
            starfield_wcs=starfield_wcs,
            n_procs=n_procs,
            processor=PUNCHImageProcessor(None, apply_mask=True, key="A"),
            handle_wrap_point=False,
            dtype=np.float32,
            target_mem_usage=target_mem_usage)
        logger.info("Ending clear starfield")
        out_data = starfield_clear.starfield - percentile_filter(starfield_clear.starfield, 5, 10)
        out_data[out_data < 0] = 0
        out_wcs = calculate_helio_wcs_from_celestial(starfield_clear.wcs,
                                                        meta.astropy_time,
                                                        starfield_clear.starfield.shape)

    # TODO - Replace uncertainty below with values folded through starfield estimation logic
    output = NDCube(data=out_data, uncertainty=StdDevUncertainty(np.sqrt(out_data)), wcs=out_wcs, meta=meta)
    output.meta.history.add_now("LEVEL3-starfield_background", "constructed starfield_bg model")

    logger.info("construct_starfield_background finished")

    if out_file is not None:
        write_ndcube_to_fits(output, filename=out_file, write_hash=False, overwrite=True)
        return None

    return [output]


@punch_task
def subtract_starfield_background_task(data_object: NDCube,
                                       before_starfield_path: str | None,
                                       after_starfield_path: str | None,
                                       is_polarized: bool = False) -> NDCube:
    """
    Subtracts a background starfield from an input data frame.

    checks the dimensions of input data frame and background starfield match and
    subtracts the background starfield from the data frame of interest.

    Parameters
    ----------
    data_object : NDCube
        A NDCube data frame to be background subtracted
    before_starfield_path : str
        path to a NDCube background starfield map centered before the observation
    after_starfield_path : str
        path to a NDCube background starfield map centered after the observation
    is_polarized : bool
        whether the data is polarized

    Returns
    -------
    NDCube
        A background starfield subtracted data frame

    """
    logger = get_run_logger()
    logger.info("subtract_starfield_background started")

    if before_starfield_path is None and after_starfield_path is None:
        output = data_object
        output.meta.history.add_now("LEVEL3-subtract_starfield_background",
                                           "starfield subtraction skipped since path is empty")
    elif before_starfield_path is None or after_starfield_path is None:
        raise InvalidDataError("subtract_starfield_background requires two input starfield models.")
    else:
        star_datacube_before = load_ndcube_from_fits(before_starfield_path)
        star_datacube_after = load_ndcube_from_fits(after_starfield_path)

        shape_before = star_datacube_before.data.shape[-2:]
        shape_after = star_datacube_after.data.shape[-2:]

        wcs_celestial_before = calculate_celestial_wcs_from_helio(star_datacube_before.wcs)
        wcs_celestial_before.wcs.cdelt[0] = wcs_celestial_before.wcs.cdelt[0] * -1

        wcs_celestial_after = calculate_celestial_wcs_from_helio(star_datacube_after.wcs)
        wcs_celestial_after.wcs.cdelt[0] = wcs_celestial_after.wcs.cdelt[0] * -1

        # TODO - Test with polarized data...
        union_wcs, union_shape = find_optimal_celestial_wcs(
            [(shape_before, wcs_celestial_before),
            (shape_after,  wcs_celestial_after)],
            auto_rotate=False, projection="CAR")

        starfield_reprojected_before = reproject_interp(
            (np.stack([star_datacube_before.data, star_datacube_before.uncertainty.array], axis=0),
            wcs_celestial_before),
            union_wcs,
            shape_out=union_shape,
            return_footprint=False)

        starfield_reprojected_after = reproject_interp(
            (np.stack([star_datacube_after.data, star_datacube_after.uncertainty.array], axis=0),
            wcs_celestial_after),
            union_wcs,
            shape_out=union_shape,
            return_footprint=False)

        starfield_before = NDCube(data=starfield_reprojected_before[0],
                                uncertainty = StdDevUncertainty(starfield_reprojected_before[1]),
                                wcs = union_wcs, meta=star_datacube_before.meta)
        starfield_after = NDCube(data=starfield_reprojected_after[0],
                                uncertainty = StdDevUncertainty(starfield_reprojected_after[1]),
                                wcs = union_wcs, meta=star_datacube_after.meta)

        starfield_data_interpolated, starfield_uncert_interpolated = interpolate_data(starfield_before,
                                                        starfield_after,
                                                        data_object.meta.datetime,
                                                        allow_extrapolation=False,
                                                        and_uncertainty=True,
                                                        infill_nans=True)
        # TODO - metadata...
        star_datacube = NDCube(data=starfield_data_interpolated,
                            uncertainty=StdDevUncertainty(starfield_uncert_interpolated),
                            wcs = union_wcs,
                            meta=star_datacube_before.meta)
        wcs_celestial = union_wcs

        original_mask = data_object.data == 0

        # TODO - Think about where to do the interpolation at this stage...
        # Is this going to require a change in the subtraction code to avoid more reprojections back and forth?
        if is_polarized:
            starfield_model_m = Starfield(np.stack((star_datacube.data[0], star_datacube.uncertainty.array[0])),
                                          wcs_celestial[0])
            subtracted_m = starfield_model_m.subtract_from_image(
                NDCube(data=np.stack((data_object.data[0], data_object.uncertainty.array[0])),
                       wcs=data_object.wcs[0],
                       meta=data_object.meta),
                handle_wrap_point=False,
                processor=PUNCHImageProcessor(layer=0, key="A"))
            starfield_model_z = Starfield(np.stack((star_datacube.data[1], star_datacube.uncertainty.array[1])),
                                          wcs_celestial[1])
            subtracted_z = starfield_model_z.subtract_from_image(
                NDCube(data=np.stack((data_object.data[1], data_object.uncertainty.array[1])),
                       wcs=data_object.wcs[1],
                       meta=data_object.meta),
                handle_wrap_point=False,
                processor=PUNCHImageProcessor(layer=1, key="A"))
            starfield_model_p = Starfield(np.stack((star_datacube.data[2], star_datacube.uncertainty.array[2])),
                                          wcs_celestial[2])
            subtracted_p = starfield_model_p.subtract_from_image(
                NDCube(data=np.stack((data_object.data[2], data_object.uncertainty.array[2])),
                       wcs=data_object.wcs[2],
                       meta=data_object.meta),
                handle_wrap_point=False,
                processor=PUNCHImageProcessor(layer=2, key="A"))

            data_object.data[...] = np.stack([subtracted_m.subtracted[0],
                                              subtracted_z.subtracted[0],
                                              subtracted_p.subtracted[0]], axis=0)
            data_object.uncertainty.array[...] = np.sqrt(data_object.uncertainty.array**2 +
                                                         np.stack([subtracted_m.subtracted[1],
                                                                   subtracted_z.subtracted[1],
                                                                   subtracted_p.subtracted[1]], axis=0)**2)
        else:
            starfield_model = Starfield(np.stack((star_datacube.data, star_datacube.uncertainty.array)), wcs_celestial)
            subtracted = starfield_model.subtract_from_image(
                NDCube(data=np.stack((data_object.data, data_object.uncertainty.array)),
                       wcs=data_object.wcs,
                       meta=data_object.meta),
                handle_wrap_point=False,
                processor=PUNCHImageProcessor(key="A"))

            data_object.data[...] = subtracted.subtracted[0]
            data_object.uncertainty.array[...] = np.sqrt(data_object.uncertainty.array**2 +
                                                         subtracted.subtracted[1]**2)

        # Reset the data to be zero in invalid regions
        data_object.data[original_mask] = 0
        data_object.data[~np.isfinite(data_object.data)] = 0

        data_object.meta.history.add_now("LEVEL3-subtract_starfield_background", "subtracted starfield background")
        output = polarize_celestial_to_solar(data_object) if is_polarized else data_object
    logger.info("subtract_starfield_background finished")

    return output


def create_empty_starfield_background(data_object: NDCube) -> np.ndarray:
    """Create an empty starfield background map."""
    return np.zeros_like(data_object.data)
