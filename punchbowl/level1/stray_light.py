import os
import pathlib
import warnings

import numpy as np
from astropy.time import Time
from ndcube import NDCube
from prefect import get_run_logger

from punchbowl.data import load_ndcube_from_fits
from punchbowl.exceptions import (
    IncorrectPolarizationStateWarning,
    IncorrectTelescopeWarning,
    InvalidDataError,
    LargeTimeDeltaWarning,
)
from punchbowl.prefect import punch_task


def _zvalue_from_index(arr, ind):  # noqa: ANN202, ANN001
    """
    Do math.

    Private helper function to work around the limitation of np.choose() by employing np.take().
    arr has to be a 3D array
    ind has to be a 2D array containing values for z-indicies to take from arr
    See: http://stackoverflow.com/a/32091712/4169585
    This is faster and more memory efficient than using the ogrid based solution with fancy indexing.
    """
    # get number of columns and rows
    _, n_cols, n_rows = arr.shape

    # get linear indices and extract elements with np.take()
    idx = n_cols*n_rows*ind + n_rows*np.arange(n_rows)[:,None] + np.arange(n_cols)
    return np.take(arr, idx)


def nan_percentile(arr: np.ndarray, q: list[float] | float) -> np.ndarray:
    """Calculate the nan percentile faster of a 3D cube."""
    # np.nanpercentile is slow so use this: https://krstn.eu/np.nanpercentile()-there-has-to-be-a-faster-way/

    # valid (non NaN) observations along the first axis
    valid_obs = np.sum(np.isfinite(arr), axis=0)
    # replace NaN with maximum
    max_val = np.nanmax(arr)
    arr[np.isnan(arr)] = max_val
    # sort - former NaNs will move to the end
    arr = np.sort(arr, axis=0)

    # loop over requested quantiles
    if isinstance(q, list):
        qs = []
        qs.extend(q)
    else:
        qs = [q]

    if len(qs) < 2:
        quant_arr = np.zeros(shape=(arr.shape[1], arr.shape[2]))
    else:
        quant_arr = np.zeros(shape=(len(qs), arr.shape[1], arr.shape[2]))

    result = []
    for i in range(len(qs)):
        quant = qs[i]
        # desired position as well as floor and ceiling of it
        k_arr = (valid_obs - 1) * (quant / 100.0)
        f_arr = np.floor(k_arr).astype(np.int32)
        c_arr = np.ceil(k_arr).astype(np.int32)
        fc_equal_k_mask = f_arr == c_arr

        # linear interpolation (like numpy percentile) takes the fractional part of desired position
        floor_val = _zvalue_from_index(arr=arr, ind=f_arr) * (c_arr - k_arr)
        ceil_val = _zvalue_from_index(arr=arr, ind=c_arr) * (k_arr - f_arr)

        quant_arr = floor_val + ceil_val
        # if floor == ceiling take floor value
        quant_arr[fc_equal_k_mask] = _zvalue_from_index(arr=arr, ind=k_arr.astype(np.int32))[fc_equal_k_mask]

        result.append(quant_arr)

    return result


def estimate_stray_light(filepaths: [str], percentile: float = 3) -> np.ndarray:
    """Estimate the fixed stray light pattern using a percentile."""
    data = np.array([load_ndcube_from_fits(path).data for path in filepaths])
    return nan_percentile(data, percentile)


@punch_task
def remove_stray_light_task(data_object: NDCube, stray_light_path: pathlib) -> NDCube:
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
    data_object : PUNCHData
        data to operate on

    stray_light_path: pathlib
        path to stray light model to apply to data

    Returns
    -------
    PUNCHData
        modified version of the input with the stray light removed

    """
    logger = get_run_logger()
    logger.info("remove_stray_light started")

    if stray_light_path is None:
        data_object.meta.history.add_now("LEVEL1-remove_stray_light", "Stray light correction skipped")
    elif not stray_light_path.exists():
        msg = f"File {stray_light_path} does not exist."
        raise InvalidDataError(msg)
    else:
        stray_light_model = load_ndcube_from_fits(stray_light_path)

        stray_light_model_date = Time(stray_light_model.meta["DATE-OBS"].value)
        observation_date = Time(data_object.meta["DATE-OBS"].value)
        if abs((stray_light_model_date - observation_date).to("day").value) > 14:
            msg=f"Calibration file {stray_light_path} contains data created greater than 2 weeks from the obsveration"
            warnings.warn(msg,LargeTimeDeltaWarning)

        if stray_light_model.meta["TELESCOP"].value != data_object.meta["TELESCOP"].value:
            msg=f"Incorrect TELESCOP value within {stray_light_path}"
            warnings.warn(msg, IncorrectTelescopeWarning)
        elif stray_light_model.meta["OBSLAYR1"].value != data_object.meta["OBSLAYR1"].value:
            msg=f"Incorrect polarization state within {stray_light_path}"
            warnings.warn(msg, IncorrectPolarizationStateWarning)
        elif stray_light_model.data.shape != data_object.data.shape:
            msg = f"Incorrect stray light function shape within {stray_light_path}"
            raise InvalidDataError(msg)
        else:
            data_object.data[:, :] -= stray_light_model.data[:, :]
            data_object.uncertainty.array[...] -= stray_light_model.data[:, :]
            data_object.meta["CALSL"] = os.path.basename(str(stray_light_path))
            data_object.meta.history.add_now("LEVEL1-remove_stray_light",
                                             f"stray light removed with {os.path.basename(str(stray_light_model))}")

    logger.info("remove_stray_light finished")
    return data_object
