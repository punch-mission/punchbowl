import warnings
from datetime import datetime

import numpy as np
from ndcube import NDCube
from prefect import get_run_logger, task

from punchbowl.data import load_ndcube_from_fits
from punchbowl.exceptions import InvalidDataError


@task
def query_f_corona_model_source(
    polarizer: str, product: str, start_datetime: datetime, end_datetime: datetime,
) -> list[str]:
    """
    Create a list of files for later F corona model generation.

    Creates a list of files based between a start date/time (start_datetime)
    and an end date/time (end_datetime) for a specified polarizer and
    PUNCH_product. The start and end times can both be input explicitly,
    individually, or derived from a mid time.

    if start_datetime (datetime object) and an end_datetime(datetime object)
    are specified a list of files between those dates is produced for the
    specified polarizer and PUNCH_product.


    Parameters
    ----------
    polarizer : string [= 'clear', '-60', '60', '0' ]
        input a string specifying the type of polarizer to search for

    product : string [= 'mosaic', 'nfi']
        product code to analyze

    start_datetime : datetime
        input a start_datetime of interest.

    end_datetime : datetime
        input a start_datetime of interest.

    Returns
    -------
    file_list : [list]
        returns a list of files over the specified period for the specified
        polarizer.

    """
    logger = get_run_logger()
    logger.info("query_f_corona_model_source started")

    # Check for polarizer input, and if valid
    if polarizer not in ["clear", "-60", "60", "0"]:
        msg = f"input polarizer expected as string: 'clear', '-60', '60', '0'. Found {polarizer}."
        raise ValueError(msg)

    # Check for PUNCH input, and if valid
    if product not in ["nfi", "mosaic"]:
        msg = f"input PUNCH_product expected as string:  'nfi', 'mosaic'. Found {product}"
        raise ValueError(msg)

    file_list = []
    # file_list=session.query(File).where(and_(File.state == "created",
    #                                           File.date_obs >= date.today() + relativedelta(months=-1))).all()

    # Check if files found in range
    if len(file_list) == 0:
        warnings.warn(f"No files found over specified dates: from: {start_datetime} to: {end_datetime}.", stacklevel=2)

    logger.info("query_f_corona_model_source finished")

    return file_list


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


@task
def construct_f_corona_background(
    data_list: list[str],
    layer: int,
) -> NDCube:
    """Build f corona background model."""
    logger = get_run_logger()
    logger.info("construct_f_corona_background started")

    # create an empty array to fill with data
    #   open the first file in the list to ge the shape of the file
    if len(data_list) == 0:
        msg = "data_list cannot be empty"
        raise ValueError(msg)

    output = load_ndcube_from_fits(data_list[0])
    output_wcs = output.wcs
    output_meta = output.meta
    output_mask = output.mask

    data_shape = output.data[layer].shape

    number_of_data_frames = len(data_list)
    data_cube = np.empty((number_of_data_frames, *data_shape), dtype=float)
    uncertainty_cube = np.empty((number_of_data_frames, *data_shape), dtype=float)
    meta_list = []

    for i, address_out in enumerate(data_list):
        data_object = load_ndcube_from_fits(address_out)
        data_cube[i, :, :] = data_object.data[layer]
        uncertainty_cube[i, :, :] = data_object.uncertainty.array[layer]
        meta_list.append(data_object.meta)


    # create an output PUNCHdata object
    f_background = np.zeros_like(data_cube[0])
    output = NDCube(f_background, wcs=output_wcs, meta=output_meta, mask=output_mask)

    logger.info("construct_f_corona_background finished")
    output.meta.history.add_now("LEVEL3-construct_f_corona_background", "constructed f corona model")

    return output


def subtract_f_corona_background(data_object: NDCube,
                                 before_f_background_model: NDCube,
                                 after_f_background_model: NDCube ) -> NDCube:
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

    before_date = before_f_background_model.meta.datetime.timestamp()
    after_date = after_f_background_model.meta.datetime.timestamp()
    observation_date = data_object.meta.datetime.timestamp()

    if before_date > observation_date:
        msg = "Before F corona model was after the observation date"
        raise InvalidDataError(msg)

    if after_date < observation_date:
        msg = "After F corona model was before the observation date"
        raise InvalidDataError(msg)

    interpolated_model = ((after_f_background_model.data - before_f_background_model.data)
                          * (observation_date - before_date) / (after_date - before_date)
                          + before_f_background_model.data)
    interpolated_model[np.isinf(data_object.uncertainty.array)] = 0

    bkg_subtracted_data = data_object.data - interpolated_model

    data_object.data[...] = bkg_subtracted_data
    data_object.uncertainty.array[:, :] -= interpolated_model
    return data_object

@task
def subtract_f_corona_background_task(observation: NDCube,
                                      before_f_background_model_path: str,
                                      after_f_background_model_path: str) -> NDCube:
    """
    Subtracts a background f corona model from an observation.

    This algorithm linearly interpolates between the before and after models.

    Parameters
    ----------
    observation : NDCube
        an observation to subtract an f corona model from

    before_f_background_model_path : str
        path to a NDCube f corona background map before the observation

    after_f_background_model_path : str
        path to a NDCube f corona background map after the observation

    Returns
    -------
    NDCube
        A background subtracted data frame

    """
    logger = get_run_logger()
    logger.info("subtract_f_corona_background started")


    before_f_corona_model = load_ndcube_from_fits(before_f_background_model_path)
    after_f_corona_model = load_ndcube_from_fits(after_f_background_model_path)

    output = subtract_f_corona_background(observation, before_f_corona_model, after_f_corona_model)
    output.meta.history.add_now("LEVEL3-subtract_f_corona_background", "subtracted f corona background")

    logger.info("subtract_f_corona_background finished")

    return output


def create_empty_f_background_model(data_object: NDCube) -> np.ndarray:
    """Create an empty background model."""
    return np.zeros_like(data_object.data)
