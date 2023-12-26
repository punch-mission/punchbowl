import numpy as np
from numpy.lib.stride_tricks import as_strided
from prefect import get_run_logger, task

from punchbowl.data import PUNCHData


def sliding_window(arr: np.ndarray, window_size: int) -> np.ndarray:
    """
    Construct a sliding window view of the array
    borrowed from: https://stackoverflow.com/questions/10996769/pixel-neighbors-in-2d-array-image-using-python
    """

    arr = np.asarray(arr)
    window_size = int(window_size)
    if arr.ndim != 2:
        raise ValueError("need 2-D input")
    if not (window_size > 0):
        raise ValueError("need a positive window size")
    shape = (arr.shape[0] - window_size + 1,
             arr.shape[1] - window_size + 1,
             window_size, window_size)
    if shape[0] <= 0:
        shape = (1, shape[1], arr.shape[0], shape[3])
    if shape[1] <= 0:
        shape = (shape[0], 1, shape[2], arr.shape[1])
    strides = (arr.shape[1]*arr.itemsize, arr.itemsize,
               arr.shape[1]*arr.itemsize, arr.itemsize)
    return as_strided(arr, shape=shape, strides=strides)


def cell_neighbors(arr: np.ndarray, i: int, j:int, window_size:int=1) -> np.ndarray:
    """
    Return d-th neighbors of cell (i, j)
    borrowed from: https://stackoverflow.com/questions/10996769/pixel-neighbors-in-2d-array-image-using-python
    """
    window = sliding_window(arr, 2*window_size+1)

    ix = np.clip(i - window_size, 0, window.shape[0]-1)
    jx = np.clip(j - window_size, 0, window.shape[1]-1)

    i0 = max(0, i - window_size - ix)
    j0 = max(0, j - window_size - jx)
    i1 = window.shape[2] - max(0, window_size - i + ix)
    j1 = window.shape[3] - max(0, window_size - j + jx)

    return window[ix, jx][i0:i1, j0:j1].ravel()


def mean_correct(data_array: np.ndarray,
                 mask_array: np.ndarray,
                 required_good_count: int = 3,
                 max_window_size: int = 10) -> np.ndarray:
    # todo: add docstring
    x_bad_pix, y_bad_pix = np.where(mask_array == 0)
    data_array[mask_array == 0] = 0
    output_data_array=data_array.copy()
    for x_i, y_i in zip(x_bad_pix, y_bad_pix):
        window_size = 1
        number_good_px=np.sum(cell_neighbors(mask_array, x_i, y_i, window_size=window_size))
        while number_good_px < required_good_count:
            window_size += 1
            number_good_px = np.sum(cell_neighbors(mask_array, x_i, y_i, window_size=window_size))
            if window_size > max_window_size:
                break
        output_data_array[x_i, y_i] = np.sum(cell_neighbors(data_array,
                                                            x_i,
                                                            y_i,
                                                            window_size=window_size))/number_good_px

    return output_data_array


def median_correct(data_array: np.ndarray,
                   mask_array: np.ndarray,
                   required_good_count: int = 3,
                   max_window_size: int = 10) -> np.ndarray:
    # todo: add docstring nd combine with mean_correct
    x_bad_pix, y_bad_pix = np.where(mask_array == 0)
    data_array[mask_array == 0] = np.nan
    output_data_array=data_array.copy()
    for x_i, y_i in zip(x_bad_pix, y_bad_pix):
        window_size = 1
        number_good_px=np.sum(cell_neighbors(mask_array, x_i, y_i, window_size=window_size))
        while number_good_px < required_good_count:
            window_size += 1
            number_good_px = np.sum(cell_neighbors(mask_array, x_i, y_i, window_size=window_size))
            if window_size > max_window_size:
                break
        output_data_array[x_i, y_i] = np.nanmedian(cell_neighbors(data_array, x_i, y_i, window_size=window_size))

    return output_data_array


@task
def remove_deficient_pixels_task(data: PUNCHData,
                                 deficient_pixel_map: PUNCHData,
                                 required_good_count: int = 3,
                                 max_window_size: int = 10,
                                 method: str = "median"
                                 ) -> PUNCHData:
    """subtracts a deficient pixel map from an input data frame.

    checks the dimensions of input data frame and map match and
    subtracts the background model from the data frame of interest.

    Parameters
    ----------
    data : PUNCHData
        A PUNCHobject data frame to be background subtracted

    deficient_pixel_map : PUNCHData
        The deficient pixels to be corrected

    required_good_count : int
        how many neighboring pixels must not be deficient to correct a pixel,
            if fewer than that many pixels are good neighbors then the box expands

    max_window_size : int
        the width of the max window

    method : str
        either "mean" or "median" depending on which measure should fill the deficient pixel


    Returns
    -------

    bkg_subtracted_data : ['punchbowl.data.PUNCHData']
        A background subtracted data frame

    # TODO: exclude data if flagged in weight array
    # TODO: update meta data with input file and version of deficient pixel map
    # TODO: output weight - update weights
    # TODO: if uncertainty object in PUNCH object is updated, then this should be updated here
    """

    logger = get_run_logger()
    logger.info("remove_deficient_pixels started")

    # todo : remove these references in favor of using the data directly
    data_array = data.data
    output_uncertainty = data.uncertainty

    deficient_pixel_array=deficient_pixel_map.data

    # check dimensions match
    if data_array.shape != deficient_pixel_array.shape:
        raise ValueError("deficient_pixel_array expects the data_object and"
                         "deficient_pixel_array arrays to have the same dimensions."
                         f"data_array dims: {data_array.shape}"
                         f"and deficient_pixel_map dims: {deficient_pixel_array.shape}")

    if method == "median":
        data_array = median_correct(data_array,
                                    deficient_pixel_array,
                                    required_good_count=required_good_count,
                                    max_window_size=max_window_size
                                    )

    elif method == "mean":
        data_array = mean_correct(data_array,
                                  deficient_pixel_array,
                                  required_good_count=required_good_count,
                                  max_window_size=max_window_size
                                  )

    else:
        raise ValueError(f"method specified must be 'mean', or 'median'. Found method={method}")

    # Set deficient pixels to infinity
    output_uncertainty.array[deficient_pixel_array == 0] = 0

    # todo: make use the duplicate_with_updates method
    output_object = data.duplicate_with_updates(data=data_array,
                                                uncertainty=output_uncertainty)

    logger.info("remove_deficient_pixels finished")
    output_object.meta.history.add_now("LEVEL1-remove_deficient_pixels", "deficient pixels removed")

    return output_object


def create_all_valid_deficient_pixel_map(data: PUNCHData) -> PUNCHData:
    mask_array = np.ones_like(data.data)
    return data.duplicate_with_updates(data=mask_array)
