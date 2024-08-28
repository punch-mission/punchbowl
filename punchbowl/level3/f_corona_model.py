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
    # TODO: Improve Query database code
    # TODO: Change placeholder output list
    # TODO: add option to have selective cadence
    # TODO: update wanings and exceptions to match standards
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

    # TODO - this place holder has to be removed
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
    method: str = "percentile",
    apply_threshold_mask: bool = True,
    threshold_mask_cutoff: float = 1.5,
    percentile_value: float = 5,
) -> NDCube:
    """
    Create a background f corona map from a series of different models.

    Creates a background f corona map (method) using a series of different
    averaging and minimization methods, values calculated across an input
    data cube of dimensions array((z,y,x)). The x,y plane should contain
    the spatial information, and the z dimension the temporal information,
    the dimension to be minimized/averaged over. The background is taken
    across the z dimension of the data cube. These include:

        'mean' - takes the average across each pixel in the z dimension
        of the data cube.

        'min' - takes the min value across each pixel in the z dimension
        of the data cube.

        'percentile' - computes the percentile of the data along the z axis
        of the data cube. Requires kwargs 'percentile_value', or the 5th
        percentile is used by default. Optional inputs include:

            percentile_value - the value to which the percentile should be
            calculated

    A threshold can be applied to reject outliers by applying
    apply_threshold_mask=1. The threshold uses the standard deviation to
    identify outliers, mask, and reject them. The masked array is used with the
    above background methods. If apply_threshold_mask is invoked a
    threshold_mask_cutoff is required, thus us the sigma-level at which to
    discard outliers. If not specified a default is 1.5.


    Parameters
    ----------
    data_list :
        list of filenames to use

    layer: int
        which layer of the cube to use

    method : [string = 'percentile', 'mean', min,]
        defines the type of background model to build. Options include

    percentile_value : [float]
        if using method='percentile', this is the value of the
        percentile the data along the z axis is calculated to.

    apply_threshold_mask : [int]
        if set outliers are masked from the data cube using the
        threshold_mask_cutoff parameter.

    threshold_mask_cutoff: [float]
        if using method='stacker', this is the sigma-level at which
        to discard outliers. The default is 1.5.

    Returns
    -------
    return output_PUNCHobject : ['punchbowl.data.PUNCHData']
        returns an array of the same dimensions as the x and y dimensions of
        the input array

    """
    # TODO: exclude data if flagged in weight array
    # TODO: pass through REAL meta data and WCS
    # TODO: create 2nd hdu with list of input files
    # TODO: add an x,y window to average over
    # TODO: needs to look at the weights (uncertainties) for trefoil images, so we don't average
    # TODO: output weight
    logger = get_run_logger()
    logger.info("construct_f_corona_background started")

    # create an empty array to fill with data
    #   open the first file in the list to ge the shape of the file
    if len(data_list) == 0:
        msg = "data_list cannot be empty"
        raise ValueError(msg)

    # TODO: replace in favor of using object directly
    output = load_ndcube_from_fits(data_list[0])
    output_wcs = output.wcs
    output_meta = output.meta
    output_mask = output.mask

    data_shape = np.shape(output.data[layer])

    number_of_data_frames = len(data_list)
    data_cube = np.empty((number_of_data_frames, *data_shape), dtype=float)

    # create an empty list for data
    meta_list = []

    for i, address_out in enumerate(data_list):
        data_object = load_ndcube_from_fits(address_out)
        data_cube[i, :, :] = data_object.data[layer]
        meta_list.append(data_object.meta)

    if apply_threshold_mask:
        # copy data arrays
        data_size = data_cube.shape
        temp_data_cube = data_cube.copy()
        data_mask = data_cube.copy()
        data_mask.fill(1)

        # Compute the total image mean
        mean_image = np.average(data_cube, axis=0)

        # Compute the standard deviation for each pixel
        sd_image = np.std(data_cube, ddof=1, axis=0)

        # Mask outliers in the cube
        for n in range(data_size[0]):
            temp_data_cube[n, :, :] = np.abs(data_cube[n, :, :] - mean_image[:, :]) / sd_image[:, :]

        # mask the data
        data_mask[temp_data_cube > threshold_mask_cutoff] = np.nan
        data_cube *= data_mask

    # calculate the background model
    if method == "percentile":
        f_background = nan_percentile(data_cube, percentile_value)
    elif method == "min":
        f_background = np.nanmin(data_cube, axis=0)
    elif method == "mean":
        f_background = np.nanmean(data_cube, axis=0)
    else:
        msg = f"Invalid f corona model supplied, method expects 'min', 'mean', or 'percentile'. Found {method}"
        raise ValueError(
            msg,
        )

    # create an output PUNCHdata object
    # TODO: the weight and wcs should come from all of the input files, not just one
    output = NDCube(f_background, wcs=output_wcs, meta=output_meta, mask=output_mask)

    logger.info("construct_f_corona_background finished")
    output.meta.history.add_now("LEVEL3-construct_f_corona_background", "constructed f corona model")

    return output


def subtract_f_corona_background(data_object: NDCube, f_background_model_array: np.ndarray) -> NDCube:
    """Subtract f corona background."""
    # check dimensions match
    if data_object.data.shape != f_background_model_array.shape:
        msg = (
            "f_background_subtraction expects the data_object and"
            "f_background arrays to have the same dimensions."
            f"data_array dims: {data_object.data.shape} and f_background_model dims: {f_background_model_array.shape}"
        )
        raise InvalidDataError(
            msg,
        )

    bkg_subtracted_data = data_object.data - f_background_model_array

    data_object.data[...] = bkg_subtracted_data
    return data_object

@task
def subtract_f_corona_background_task(data_object: NDCube,
                                      f_background_model_path: str | None) -> NDCube:
    """
    Subtracts a background f corona model from an input data frame.

    checks the dimensions of input data frame and background model match and
    subtracts the background model from the data frame of interest.

    Parameters
    ----------
    data_object : punchbowl.data.PUNCHData
        A PUNCHobject data frame to be background subtracted

    f_background_model_path : str
        path to a PUNCHobject background map

    Returns
    -------
    bkg_subtracted_data : ['punchbowl.data.PUNCHData']
        A background subtracted data frame

    """
    # TODO: exclude data if flagged in weight array
    # TODO: pass through REAL meta data and WCS
    # TODO: create 2nd hdu with list of input files
    # TODO: needs to look at the weights (uncertainties) for trefoil images, so we don't average
    # TODO: output weight - combine weights
    logger = get_run_logger()
    logger.info("subtract_f_corona_background started")

    if f_background_model_path is None:
        output = data_object
        output.meta.history.add_now("LEVEL3-fcorona-subtraction",
                                           "F corona subtraction skipped since path is empty")
    else:
        f_data_array = load_ndcube_from_fits(f_background_model_path).data
        output = subtract_f_corona_background(data_object, f_data_array)
        output.meta.history.add_now("LEVEL3-subtract_f_corona_background", "subtracted f corona background")

    logger.info("subtract_f_corona_background finished")

    return output


def create_empty_f_background_model(data_object: NDCube) -> np.ndarray:
    """Create an empty background model."""
    return np.zeros_like(data_object.data)
