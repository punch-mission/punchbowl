from typing import Optional
from datetime import datetime
from prefect import task, get_run_logger
import numpy as np
from numpy.lib.stride_tricks import as_strided

from punchbowl.data import PUNCHData


def sliding_window(arr, window_size):
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

def cell_neighbors(arr, i, j, window_size):
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

    return window[ix, jx][i0:i1,j0:j1].ravel()

def mean_example(data_array, mask_array, required_good_count=3, max_window_size=10):
    x_bad_pix,y_bad_pix=np.where(mask_array==0)
    data_array[mask_array==0] = 0
    output_data_array=data_array.copy()
    for x_i, y_i in zip(x_bad_pix,y_bad_pix):
        window_size=1
        number_good_px=np.sum(cell_neighbors(mask_array, x_i, y_i, d=window_size))
        while number_good_px < required_good_count:
            window_size+=1
            number_good_px=np.sum(cell_neighbors(mask_array, x_i, y_i, d=window_size))
            if window_size > max_window_size:
                break
        output_data_array[x_i, y_i]=np.sum(cell_neighbors(data_array, x_i, y_i, d=window_size))/number_good_px
    
    return output_data_array


def median_example(data_array, mask_array, required_good_count=3, max_window_size=10):
    x_bad_pix,y_bad_pix=np.where(mask_array==0)
    data_array[mask_array==0]=np.nan
    output_data_array=data_array.copy()
    for x_i, y_i in zip(x_bad_pix,y_bad_pix):
        window_size=1
        number_good_px=np.sum(cell_neighbors(mask_array, x_i, y_i, d=window_size))
        while number_good_px < required_good_count:
            window_size+=1
            number_good_px=np.sum(cell_neighbors(mask_array, x_i, y_i, d=window_size))
            if window_size > max_window_size:
                break
        output_data_array[x_i, y_i]=np.nanmedian(cell_neighbors(data_array, x_i, y_i, d=window_size))

    return output_data_array


@task
def remove_deficient_pixels(data_object: PUNCHData, 
                            deficient_pixel_map: PUNCHData,
                            required_good_count: int= 3, 
                            max_window_size: int= 10,
                            method: str= 'median'
                            ) -> PUNCHData:
    """subtracts a deficient pixel map from an input data frame.
        
    checks the dimensions of input data frame and map match and
    subtracts the background model from the data frame of interest.
    
    Parameters
    ----------
    data : PUNCHData
        A PUNCHobject data frame to be background subtracted
        
    deficient_pixel_map : PUNCHData
        A deficient_pixel map

    Returns
    -------

    bkg_subtracted_data : ['punchbowl.data.PUNCHData']
        A background subtracted data frame

    TODO
    ----
    
    # TODO: exclude data if flagged in weight array
    # TODO: update meta data with input file and version of deficient pixel map
    # TODO: output weight - update weights
    """

    logger = get_run_logger()
    logger.info("remove_deficient_pixels started")
    
    data_array=data_object.data
    output_wcs=data_object.wcs
    output_meta=data_object.meta
    output_uncertainty=data_object.uncertainty
    output_mask=data_object.mask
    
    deficient_pixel_array=deficient_pixel_map.data

    # check dimensions match
    if data_array.shape != deficient_pixel_array.shape:
        raise Exception("deficient_pixel_array expects the data_object and"
                         "deficient_pixel_array arrays to have the same dimensions." 
                         f"data_array dims: {data_array.shape} and deficient_pixel_map dims: {deficient_pixel_array.shape}")
            
    if method == 'median':
        data_array = median_example(data_array, 
                                    deficient_pixel_array, 
                                    required_good_count=required_good_count,
                                    max_window_size=max_window_size
                                    )

    elif method == 'mean':
        data_array = mean_example(data_array, 
                                  deficient_pixel_array, 
                                  required_good_count=required_good_count,
                                  max_window_size=max_window_size
                                  )

    else:
        raise Exception(f"method specified must be 'mean', or 'median'. Found method={method}")


    output_uncertainty[deficient_pixel_array==1] = np.inf
    
    output_PUNCHobject=PUNCHData(data_array, 
                                wcs=output_wcs, 
                                uncertainty=output_uncertainty,
                                meta=output_meta, 
                                mask=output_mask)

    logger.info("remove_deficient_pixels finished")
    output_PUNCHobject.meta.history.add_now("LEVEL1-remove_deficient_pixels", "deficient pixels removed")

    return output_PUNCHobject



