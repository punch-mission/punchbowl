import warnings
from typing import List, Optional
from datetime import datetime

import numpy as np
from prefect import get_run_logger, task

from punchbowl.data import PUNCHData
from punchbowl.exceptions import InvalidDataError


@task
def query_f_corona_model_source(
    polarizer: str, product: str, start_datetime: datetime, end_datetime: datetime
) -> List[str]:
    """Creates a list of files based between a start date/time (start_datetime)
    and an end date/time (end_datetime) for a specifed polarizer and product
    type.

    Creates a list of files based between a start date/time (start_datetime)
    and an end date/time (end_datetime) for a specifed polarizer and
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

    start_datetime : datetime
        input a start_datetime of interest.

    end_datetime : datetime
        input a start_datetime of interest.

    Returns
    -------
    file_list : [list]
        returns a list of files over the specified period for the specified
        polarizer.

    TODO
    ----
    # TODO: Improve Query database code
    # TODO: Change placeholder output list
    # TODO: add option to have selective cadence
    # TODO: update wanings and exceptions to match standards

    """
    logger = get_run_logger()
    logger.info("query_f_corona_model_source started")

    # Check for polarizer input, and if valid
    if polarizer not in ["clear", "-60", "60", "0"]:
        raise ValueError(f"input polarizer expected as string: 'clear', '-60', '60', '0'. Found {polarizer}.")

    # Check for PUNCH input, and if valid
    if product not in ["nfi", "mosaic"]:
        raise ValueError(f"input PUNCH_product expected as string:  'nfi', 'mosaic'. Found {product}")

    # TODO - this place holder has to be removed
    file_list = []
    # file_list=session.query(File).where(and_(File.state == "created",
    #                                           File.date_obs >= date.today() + relativedelta(months=-1))).all()

    # Check if files found in range
    if len(file_list) == 0:
        warnings.warn(f"No files found over specified dates: from: {start_datetime} to: {end_datetime}.", stacklevel=2)

    logger.info("query_f_corona_model_source finished")

    return file_list


@task
def construct_f_corona_background(
    data_list: List[str],
    method: str = "percentile",
    apply_threshold_mask: bool = True,
    threshold_mask_cutoff: float = 1.5,
    percentile_value: float = 5,
) -> PUNCHData:
    """Creates a background f corona map from a series of different models.

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

    # TODO: exclude data if flagged in weight array
    # TODO: pass through REAL meta data and WCS
    # TODO: create 2nd hdu with list of input files
    # TODO: add an x,y window to average over
    # TODO: needs to look at the weights (uncertainties) for trefoil images, so we don't average
    # TODO: output weight
    """
    logger = get_run_logger()
    logger.info("construct_f_corona_background started")

    # create punch data cube from input list
    # data_cube = np.stack([PUNCHData(PUNCHData.from_fits(address_out)).data for address_out in data_list])

    # create an empty array to fill with data
    #   open the first file in the list to ge the shape of the file
    if len(data_list) == 0:
        raise ValueError("data_list cannot be empty")

    # todo: replace in favor of using object directly
    output = PUNCHData.from_fits(data_list[0])
    output_wcs = output.wcs
    output_meta = output.meta
    # output_uncertainty=shape_PUNCHobject.uncertainty
    output_mask = output.mask

    data_shape = np.shape(output.data)

    number_of_data_frames = len(data_list)
    data_cube = np.empty((number_of_data_frames, *data_shape), dtype=float)

    # create an empty list for data
    meta_list = []

    for i, address_out in enumerate(data_list):
        data_object = PUNCHData.from_fits(address_out)
        data_cube[i, :, :] = data_object.data
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
            temp_data_cube[n, :, :] = np.abs(data_cube[n, :, :].astype(int) - mean_image[:, :]) / sd_image[:, :]

        # make int
        temp_data_cube = temp_data_cube.astype(int)

        # mask the data
        data_mask[temp_data_cube > threshold_mask_cutoff] = np.nan
        data_cube *= data_mask

    # calculate the background model

    # 'percentile' model
    if method == "percentile":
        f_background = np.percentile(data_cube, percentile_value, axis=0)

    # 'min' model
    elif method == "min":
        f_background = np.min(data_cube, axis=0)

    # 'mean' model
    elif method == "mean":
        f_background = np.mean(data_cube, axis=0)

    else:
        raise ValueError(
            "invalid f corona model supplied, method expects 'min', 'mean', or 'percentile'. " f"Found {method}"
        )

    # create an output PUNCHdata object
    # TODO: the weight and wcs should come from all of the input files, not just one
    output = PUNCHData(f_background, wcs=output_wcs, meta=output_meta, mask=output_mask)

    logger.info("construct_f_corona_background finished")
    output.meta.history.add_now("LEVEL3-construct_f_corona_background", "constructed f corona model")

    return output


@task
def subtract_f_corona_background_task(data_object: PUNCHData,
                                      f_background_model_path: Optional[str]) -> PUNCHData:
    """subtracts a background f corona model from an input data frame.

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

    TODO
    ----

    # TODO: exclude data if flagged in weight array
    # TODO: pass through REAL meta data and WCS
    # TODO: create 2nd hdu with list of input files
    # TODO: needs to look at the weights (uncertainties) for trefoil images, so we don't average
    # TODO: output weight - combine weights
    """
    logger = get_run_logger()
    logger.info("subtract_f_corona_background started")

    if f_background_model_path is None:
        f_data_array = create_empty_f_background_model(data_object)
    else:
        f_data_array = PUNCHData.from_fits(f_background_model_path).data

    data_array = data_object.data
    output_wcs = data_object.wcs
    output_meta = data_object.meta
    # output_uncertainty=shape_PUNCHobject.uncertainty
    output_mask = data_object.mask

    # check dimensions match
    if data_array.shape != f_data_array.shape:
        raise InvalidDataError(
            "f_background_subtraction expects the data_object and"
            "f_background arrays to have the same dimensions."
            f"data_array dims: {data_array.shape} and f_background_model dims: {f_data_array.shape}"
        )

    bkg_subtracted_data = data_array - f_data_array

    output = PUNCHData(bkg_subtracted_data, wcs=output_wcs, meta=output_meta, mask=output_mask)

    logger.info("subtract_f_corona_background finished")
    output.meta.history.add_now("LEVEL3-subtract_f_corona_background", "subtracted f corona background")

    return output


def create_empty_f_background_model(data_object: PUNCHData) -> np.ndarray:
    return np.zeros_like(data_object.data)
