import os
import pathlib
import warnings
from datetime import UTC, datetime

import numba
import numpy as np
from dateutil.parser import parse as parse_datetime
from ndcube import NDCube
from prefect import get_run_logger
from scipy.special import erfinv

from punchbowl.data import NormalizedMetadata, load_ndcube_from_fits
from punchbowl.exceptions import (
    CantInterpolateWarning,
    IncorrectPolarizationStateError,
    IncorrectTelescopeError,
    InvalidDataError,
)
from punchbowl.prefect import punch_flow, punch_task
from punchbowl.util import average_datetime, interpolate_data, parallel_sort_first_axis


@punch_flow
def estimate_stray_light(filepaths: list[str],
                         percentile: float = 1,
                         do_uncertainty: bool = True,
                         reference_time: datetime | str | None = None,
                         exclude_percentile: float = 50,
                         erfinv_scale: float = 0.75,
                         num_workers: int | None = None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Estimate the fixed stray light pattern using a percentile."""
    logger = get_run_logger()
    logger.info(f"Running with {len(filepaths)} input files")
    if isinstance(reference_time, str):
        reference_time = datetime.strptime(reference_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    data = None
    uncertainty = None
    date_obses = []
    for i, path in enumerate(sorted(filepaths)):
        try:
            cube = load_ndcube_from_fits(path, include_provenance=False, include_uncertainty=do_uncertainty)
        except:
            logger.warning(f"Error reading {path}")
            raise
        date_obses.append(cube.meta.datetime)
        if data is None:
            data = np.empty((len(filepaths), *cube.data.shape))
        data[i] = cube.data
        if do_uncertainty:
            if uncertainty is None:
                uncertainty = np.zeros_like(cube.data)
            if cube.uncertainty is not None:
                # The final uncertainty is sqrt(sum(square(input uncertainties))), so we accumulate the squares here
                uncertainty += cube.uncertainty.array ** 2

    logger.info(f"Images loaded; they span {min(date_obses).strftime('%Y-%m-%dT%H:%M:%S')} to "
                f"{max(date_obses).strftime('%Y-%m-%dT%H:%M:%S')}")

    if num_workers:
        numba.config.NUMBA_NUM_THREADS = num_workers

    parallel_sort_first_axis(data, inplace=True)

    index_exclude = np.floor(len(filepaths) * exclude_percentile / 100).astype(int)
    index_percentile = np.floor(len(filepaths) * percentile / 100).astype(int)
    stray_light_estimate = data[index_percentile, :, :]

    stray_light_std = np.std(data[0:index_exclude, :, :], axis=0)

    sigma_offset = -1 * erfinv((-1 + percentile / 50) * erfinv_scale)

    stray_light_estimate2 = stray_light_estimate + sigma_offset * stray_light_std

    if do_uncertainty:
        uncertainty = np.sqrt(uncertainty) / len(filepaths) if do_uncertainty else None

    out_type = "S" + cube.meta.product_code[1:]
    meta = NormalizedMetadata.load_template(out_type, "1")
    meta["DATE-AVG"] = average_datetime(date_obses).strftime("%Y-%m-%dT%H:%M:%S")
    meta["DATE-OBS"] = reference_time.strftime("%Y-%m-%dT%H:%M:%S") if reference_time else meta["DATE-AVG"].value
    meta["DATE-BEG"] = min(date_obses).strftime("%Y-%m-%dT%H:%M:%S")
    meta["DATE-END"] = max(date_obses).strftime("%Y-%m-%dT%H:%M:%S")
    meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")
    meta.history.add_now("stray light",
                         f"Generated with {len(filepaths)} files running from "
                         f"{min(date_obses).strftime('%Y-%m-%dT%H:%M:%S')} to "
                         f"{max(date_obses).strftime('%Y-%m-%dT%H:%M:%S')}")
    meta["FILEVRSN"] = cube.meta["FILEVRSN"].value

    # Let's put in a valid, representative WCS, with the right scale and pointing, etc. But let's set the rotation to
    # zero---the rotation value is meaningless, so it should be an obvious filler value
    wcs = cube.wcs
    wcs.wcs.pc = np.eye(2)
    out_cube = NDCube(data=stray_light_estimate2, meta=meta, wcs=wcs, uncertainty=uncertainty)

    return [out_cube]


@punch_task
def remove_stray_light_task(data_object: NDCube, #noqa: C901
                            stray_light_before_path: pathlib.Path | str | NDCube,
                            stray_light_after_path: pathlib.Path | str | NDCube) -> NDCube:
    """
    Prefect task to remove stray light from an image.

    Stray light is light in an optical system which was not intended in the
    design.

    The PUNCH instrument stray light will be mapped periodically as part of the
    ongoing in-flight calibration effort. The stray light maps will be
    generated directly from the L0 and L1 science data. Separating instrumental
    stray light from the F-corona. This has been demonstrated with SOHO/LASCO
    and with STEREO/COR2 observations. It requires an instrumental roll to hold
    the stray light pattern fixed while the F-corona rotates in the field of
    view. PUNCH orbital rolls will be used to create similar effects.

    Uncertainty across the image plane is calculated using a known stray light
    model and the difference between the calculated stray light and the ground
    truth. The uncertainty is convolved with the input uncertainty layer to
    produce the output uncertainty layer.


    Parameters
    ----------
    data_object : NDCube
        data to operate on

    stray_light_before_path: pathlib
        path to stray light model before observation to apply to data

    stray_light_after_path: pathlib
        path to stray light model after observation to apply to data

    Returns
    -------
    NDCube
        modified version of the input with the stray light removed

    """
    if stray_light_before_path is None or stray_light_after_path is None:
        data_object.meta.history.add_now("LEVEL1-remove_stray_light", "Stray light correction skipped")
        return data_object

    if isinstance(stray_light_before_path, NDCube):
        stray_light_before_model = stray_light_before_path
        stray_light_before_path = stray_light_before_model.meta["FILENAME"].value
    else:
        stray_light_before_path = pathlib.Path(stray_light_before_path)
        if not stray_light_before_path.exists():
            msg = f"File {stray_light_before_path} does not exist."
            raise InvalidDataError(msg)
        stray_light_before_model = load_ndcube_from_fits(stray_light_before_path)
        stray_light_before_path = stray_light_before_model.meta["FILENAME"].value

    if isinstance(stray_light_after_path, NDCube):
        stray_light_after_model = stray_light_after_path
        stray_light_after_path = stray_light_after_model.meta["FILENAME"].value
    else:
        stray_light_after_path = pathlib.Path(stray_light_after_path)
        if not stray_light_after_path.exists():
            msg = f"File {stray_light_after_path} does not exist."
            raise InvalidDataError(msg)
        stray_light_after_model = load_ndcube_from_fits(stray_light_after_path)

    for model in stray_light_before_model, stray_light_after_model:
        if model.meta["TELESCOP"].value != data_object.meta["TELESCOP"].value:
            msg=f"Incorrect TELESCOP value within {model['FILENAME'].value}"
            raise IncorrectTelescopeError(msg)
        if model.meta["OBSLAYR1"].value != data_object.meta["OBSLAYR1"].value:
            msg=f"Incorrect polarization state within {model['FILENAME'].value}"
            raise IncorrectPolarizationStateError(msg)
        if model.data.shape != data_object.data.shape:
            msg = f"Incorrect stray light function shape within {model['FILENAME'].value}"
            raise InvalidDataError(msg)

    # For the quickpunch case, our stray light models run right up to the current time, with their DATE-OBS likely days
    # in the past. It feels reckless to interpolate the six-hour variation in the model over several days, so let's
    # instead interpolate using the nearst of DATE-BEG, DATE-AVG, or DATE-END. (DATE-BEG will be the best choice when
    # reprocessing.)
    delta_dateavg = abs(parse_datetime(stray_light_before_model.meta["DATE-AVG"].value + " UTC")
                        - data_object.meta.datetime)
    delta_datebeg = abs(parse_datetime(stray_light_before_model.meta["DATE-BEG"].value + " UTC")
                        - data_object.meta.datetime)
    delta_dateend = abs(parse_datetime(stray_light_before_model.meta["DATE-END"].value + " UTC")
                        - data_object.meta.datetime)

    closest = min(delta_datebeg, delta_dateavg, delta_dateend)
    if closest is delta_datebeg:
        time_key = "DATE-BEG"
    elif closest is delta_dateavg:
        time_key = "DATE-AVG"
    else:
        time_key = "DATE-END"

    if stray_light_before_model.meta[time_key].value == stray_light_after_model.meta[time_key].value:
        warnings.warn(
            "Timestamps are identical for the stray light models; can't inter/extrapolate", CantInterpolateWarning)
        stray_light_model = stray_light_before_model.data
    else:
        stray_light_model = interpolate_data(stray_light_before_model,
                                             stray_light_after_model,
                                             data_object.meta.datetime,
                                             time_key=time_key,
                                             allow_extrapolation=True)
    data_object.data[:, :] -= stray_light_model
    uncertainty = 0
    # TODO: when we have real uncertainties, use them
    # uncertainty = stray_light_model.uncertainty.array # noqa: ERA001
    data_object.uncertainty.array[...] = np.sqrt(data_object.uncertainty.array**2 + uncertainty**2)
    data_object.meta.history.add_now("LEVEL1-remove_stray_light",
                                     f"stray light removed with {os.path.basename(str(stray_light_before_path))} "
                                     f"and {os.path.basename(str(stray_light_after_path))}")
    return data_object
