import os
import multiprocessing
from datetime import UTC, datetime
from concurrent.futures import ProcessPoolExecutor

import astropy
import numba
import numpy as np
from astropy.nddata import StdDevUncertainty
from dateutil.parser import parse as parse_datetime_str
from scipy.interpolate import griddata

from punchbowl.data import NormalizedMetadata
from punchbowl.data.punch_io import load_ndcube_from_fits
from punchbowl.data.punchcube import PUNCHCube
from punchbowl.data.wcs import load_trefoil_wcs
from punchbowl.exceptions import InvalidDataError
from punchbowl.prefect import get_logger, punch_flow, punch_task
from punchbowl.util import ShmPickleableNDArray, average_datetime, interpolate_data, nan_percentile


def model_fcorona_for_cube(cube: np.ndarray) -> np.ndarray:
    """
    Model the F corona given a list of times and a corresponding data cube.

    Parameters
    ----------
    xt : np.ndarray
        Unused
    reference_xt: float
        Unused
    cube : np.ndarray
        observation array
    args : list
        Kept for signature compatibility
    kwargs : dict
        Kept for signature compatibility

    Returns
    -------
    np.ndarray
        The F-corona model at the central point in time. If return_full_curves is True, this is
        instead the F-corona model at all points in time covered by the data cube
    None
        Nothing

    """
    return nan_percentile(cube, 3)


def fill_nans_with_interpolation(image: np.ndarray) -> np.ndarray:
    """Fill NaN values in an image using interpolation."""
    mask = np.isnan(image)
    x, y = np.where(~mask)
    known_values = image[~mask]

    grid_x, grid_y = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    return griddata((x, y), known_values, (grid_x, grid_y), method="cubic")


def _load_file(path: str, data_destination: ShmPickleableNDArray) -> tuple[np.ndarray, datetime, str]:
    data_destination[:] = np.nan
    try:
        cube = load_ndcube_from_fits(path, include_provenance=False, dtype=np.float32)
    except Exception as e:  # noqa: BLE001
        return str(e)

    if "CROPX1" in cube.meta:
        cropx = cube.meta["CROPX1"].value, cube.meta["CROPX2"].value
        cropy = cube.meta["CROPY1"].value, cube.meta["CROPY2"].value
    else:  # this is likely quickpunch since it doesn't have crop natively implemented now
        cropx = 0, 4096
        cropy = 0, 4096

    data_destination[:, cropy[0]:cropy[1], cropx[0]:cropx[1]] = (
        np.where(np.isfinite(cube.uncertainty.array), cube.data, np.nan)
    )
    np.nan_to_num(cube.uncertainty.array, nan=0, posinf=0, neginf=0, copy=False)
    # Square the array in-place
    cube.uncertainty.array *= cube.uncertainty.array
    uncert = np.zeros(data_destination.shape, dtype=np.float32)
    uncert[..., cropy[0]:cropy[1], cropx[0]:cropx[1]] = cube.uncertainty.array
    return uncert.squeeze(), cube.meta.datetime, cube.meta["OBSCODE"].value


@punch_flow(log_prints=True)
def construct_f_corona_model(filenames: list[str], # noqa: C901
                             reference_time: str | None = None,
                             num_workers: int = 8,
                             num_loaders: int | None = None,
                             fill_nans: bool = False,
                             polarized: bool = False) -> list[PUNCHCube]:
    """Construct a full F corona model."""
    numba.set_num_threads(num_workers)
    logger = get_logger()

    if reference_time is None:
        reference_time = datetime.now(UTC)
    elif isinstance(reference_time, str):
        reference_time = parse_datetime_str(reference_time)

    trefoil_wcs, trefoil_shape = load_trefoil_wcs()

    logger.info("construct_f_corona_background started")

    if len(filenames) == 0:
        msg = "Require at least one input file"
        raise ValueError(msg)

    filenames.sort()

    number_of_data_frames = len(filenames)

    uncertainty = np.zeros((3, *trefoil_shape) if polarized else trefoil_shape)
    sample_counts = np.zeros((3 if polarized else 1, *trefoil_shape) , dtype=int)
    data_cube = ShmPickleableNDArray((number_of_data_frames, 3 if polarized else 1, *trefoil_shape), dtype=np.float32)

    logger.info("beginning data loading")
    dates = []
    n_failed = 0
    ctx = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(num_loaders, mp_context=ctx) as pool:
        for i, result in enumerate(pool.map(_load_file, filenames, data_cube)):
            if isinstance(result, str):
                logger.warning(f"Loading {filenames[i]} failed")
                logger.warning(result)
                n_failed += 1
                if n_failed > 0.05 * len(filenames):
                    raise RuntimeError(f"{n_failed} files failed to load, stopping")
                continue
            this_uncertainty, date, obscode = result
            dates.append(date)

            sample_counts += this_uncertainty != 0
            uncertainty += this_uncertainty

            if (i + 1) % 50 == 0:
                logger.info(f"Loaded {i+1}/{len(filenames)} files")

    logger.info(f"end of data loading, saw {n_failed} failures")

    models = []
    for i in range(data_cube.shape[1]):
        model_fcorona = model_fcorona_for_cube(data_cube[:, i])
        model_fcorona[sample_counts[i] == 0] = np.nan
        if fill_nans:
            model_fcorona = fill_nans_with_interpolation(model_fcorona)
        models.append(model_fcorona)

    uncertainty = np.sqrt(uncertainty) / sample_counts

    if polarized:
        output_data = np.stack(models, axis=0)
        meta = NormalizedMetadata.load_template("PF" + obscode, "3")
        trefoil_wcs = astropy.wcs.utils.add_stokes_axis_to_wcs(trefoil_wcs, 2)
    else:
        output_data = models[0]
        meta = NormalizedMetadata.load_template("CF" + obscode, "3")

    meta.provenance = sorted([os.path.basename(f) for f in filenames])

    meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    meta["DATE-AVG"] = average_datetime(dates).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    meta["DATE-OBS"] = reference_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    meta["DATE-BEG"] = min(dates).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    meta["DATE-END"] = max(dates).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    output_cube = PUNCHCube(data=output_data,
                         meta=meta,
                         wcs=trefoil_wcs,
                         uncertainty=StdDevUncertainty(uncertainty))

    return [output_cube]

def subtract_f_corona_background(data_object: PUNCHCube,
                                 before_f_background_model: PUNCHCube,
                                 after_f_background_model: PUNCHCube,
                                 allow_extrapolation: bool = False) -> PUNCHCube:
    """Subtract f corona background."""
    # check dimensions match
    if data_object.data.shape != before_f_background_model.data.shape:
        msg = (
            "f_background_subtraction expects the data_object and"
            "f_background arrays to have the same dimensions."
            f"data_array dims: {data_object.data.shape} "
            f"and before_f_background_model dims: {before_f_background_model.data.shape}"
        )
        raise InvalidDataError(
            msg,
        )

    if data_object.data.shape != after_f_background_model.data.shape:
        msg = (
            "f_background_subtraction expects the data_object and"
            "f_background arrays to have the same dimensions."
            f"data_array dims: {data_object.data.shape} "
            f"and after_f_background_model dims: {after_f_background_model.data.shape}"
        )
        raise InvalidDataError(
            msg,
        )

    interpolated_model, interpolated_uncertainty = interpolate_data(
            before_f_background_model,
            after_f_background_model,
            data_object.meta.datetime,
            allow_extrapolation=allow_extrapolation,
            and_uncertainty=True)

    interpolated_model[(data_object.data == 0) & np.isinf(data_object.uncertainty.array)] = 0

    original_mask = (data_object.data == 0) * np.isinf(data_object.uncertainty.array)
    data_object.data[...] -= interpolated_model
    data_object.data[original_mask] = 0
    data_object.uncertainty.array[...] = np.sqrt(data_object.uncertainty.array**2 + interpolated_uncertainty**2)
    return data_object

@punch_task
def subtract_f_corona_background_task(observation: PUNCHCube,  #noqa: C901
                                      before_f_background_models: list[PUNCHCube | str],
                                      after_f_background_models: list[PUNCHCube | str],
                                      allow_extrapolation: bool = False) -> PUNCHCube:
    """
    Subtracts a background f corona model from an observation.

    This algorithm linearly interpolates between the before and after models.

    Parameters
    ----------
    observation : PUNCHCube
        an observation to subtract an f corona model from

    before_f_background_models : list[PUNCHCube | str]
        PUNCHCube f corona background maps before the observations

    after_f_background_models : list[PUNCHCube | str]
        PUNCHCube f corona background maps after the observations

    allow_extrapolation : bool
        If true, allow extrapolation beyond the time range spanned by the two F corona models

    Returns
    -------
    PUNCHCube
        A background subtracted data frame

    """
    logger = get_logger()
    logger.info("subtract_f_corona_background started")

    before_f_background_models = [load_ndcube_from_fits(f) if isinstance(f, str) else f
                                  for f in before_f_background_models]
    after_f_background_models = [load_ndcube_from_fits(f) if isinstance(f, str) else f
                                  for f in after_f_background_models]

    for model in before_f_background_models:
        if model.meta["OBSCODE"].value != observation.meta["OBSCODE"].value:
            continue
        if observation.meta["TYPECODE"].value[1] == "R" and model.meta["TYPECODE"].value[0] == "C":
            before_model = model
            break
        if observation.meta["TYPECODE"].value[1] == "P" and model.meta["TYPECODE"].value[0] == "P":
            before_model = model
            break
        if observation.meta["TYPECODE"].value[1] == "Q" and model.meta["TYPECODE"].value[0] == "C":
            before_model = model
            break
    else:
        raise RuntimeError(f"Could not find before model for {observation.meta['FILENAME']}")

    for model in after_f_background_models:
        if model.meta["OBSCODE"].value != observation.meta["OBSCODE"].value:
            continue
        if observation.meta["TYPECODE"].value[1] == "R" and model.meta["TYPECODE"].value[0] == "C":
            after_model = model
            break
        if observation.meta["TYPECODE"].value[1] == "P" and model.meta["TYPECODE"].value[0] == "P":
            after_model = model
            break
        if observation.meta["TYPECODE"].value[1] == "Q" and model.meta["TYPECODE"].value[0] == "C":
            after_model = model
            break
    else:
        raise RuntimeError(f"Could not find after model for {observation.meta['FILENAME']}")

    output = subtract_f_corona_background(observation, before_model, after_model,
                                          allow_extrapolation=allow_extrapolation)

    output.meta.history.add_now("LEVEL3-subtract_f_corona_background", "subtracted f corona background")

    logger.info("subtract_f_corona_background finished")

    return output


def create_empty_f_background_model(data_object: PUNCHCube) -> np.ndarray:
    """Create an empty background model."""
    return np.zeros_like(data_object.data)
