import os
from datetime import datetime

import numpy as np
from ndcube import NDCube

from punchbowl.data import load_ndcube_from_fits, write_ndcube_to_fits
from punchbowl.prefect import punch_task


def validate_image_is_square(image: np.ndarray) -> None:
    """Check that the input array is square."""
    if not isinstance(image, np.ndarray):
        msg = f"Image must be of type np.ndarray. Found: {type(image)}."
        raise TypeError(msg)
    if len(image.shape) != 2:
        msg = f"Image must be a 2-D array. Input has {len(image.shape)} dimensions."
        raise ValueError(msg)
    if not np.equal(*image.shape):
        msg = f"Image must be square. Found: {image.shape}."
        raise ValueError(msg)


@punch_task
def output_image_task(data: NDCube, output_filename: str) -> None:
    """
    Prefect task to write an image to disk.

    Parameters
    ----------
    data : NDCube
        data that is to be written
    output_filename : str
        where to write the file out

    Returns
    -------
    None

    """
    output_dir = os.path.dirname(output_filename)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    write_ndcube_to_fits(data, output_filename)


@punch_task
def load_image_task(input_filename: str) -> NDCube:
    """
    Prefect task to load data for processing.

    Parameters
    ----------
    input_filename : str
        path to file to load

    Returns
    -------
    NDCube
        loaded version of the image

    """
    return load_ndcube_from_fits(input_filename)


def average_datetime(datetimes: list[datetime]) -> datetime:
    """Compute average datetime from a list of datetimes."""
    timestamps = [dt.timestamp() for dt in datetimes]
    average_timestamp = sum(timestamps) / len(timestamps)
    return datetime.fromtimestamp(average_timestamp)  # noqa: DTZ006


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
    is_good = np.isfinite(arr)
    n_valid_obs = np.sum(is_good, axis=0)
    # replace NaN with maximum
    arr[~is_good] = np.nanmax(arr)
    # sort - former NaNs will move to the end
    arr = np.sort(arr, axis=0)

    # loop over requested quantiles
    qs = [q] if isinstance(q, float | int) else q

    result = np.empty((len(qs), *arr.shape[1:]))
    for i, quant in enumerate(qs):
        # desired position as well as floor and ceiling of it
        k_arr = (n_valid_obs - 1) * (quant / 100)
        f_arr = np.floor(k_arr).astype(np.int32)
        c_arr = np.ceil(k_arr).astype(np.int32)
        fc_equal_k_mask = f_arr == c_arr

        # linear interpolation (like numpy percentile) takes the fractional part of desired position
        floor_val = _zvalue_from_index(arr=arr, ind=f_arr) * (c_arr - k_arr)
        ceil_val = _zvalue_from_index(arr=arr, ind=c_arr) * (k_arr - f_arr)

        quant_arr = floor_val + ceil_val
        # if floor == ceiling take floor value
        quant_arr[fc_equal_k_mask] = _zvalue_from_index(arr=arr, ind=k_arr.astype(np.int32))[fc_equal_k_mask]

        result[i] = quant_arr

    result[:, n_valid_obs == 0] = np.nan

    return result
