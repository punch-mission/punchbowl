# Core Python imports
from typing import Optional, Tuple
from datetime import datetime

# Third party imports
import numpy as np

import reproject

from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS

from prefect import task, get_run_logger
import astropy.units as u

# Punchbowl imports
from punchbowl.data import PUNCHData

# core reprojection function
def reproject_array(input_array: np.ndarray, 
                    input_wcs: WCS, 
                    output_wcs: WCS,
                    output_shape: tuple) -> np.ndarray:
    """Core reprojection function

    Core reprojection function of the PUNCH mosaic generation module.
        With an input data array and corresponding WCS object, the function 
        performs a reprojection into the output WCS object system, along with 
        a specified pixel size for the output array. This utilizes the adaptive 
        reprojection routine implemented in the reprojection astropy package.

    Parameters
    ----------
    input_array
        input array to be reprojected
    input_wcs
        astropy WCS object describing the input array
    output_wcs
        astropy WCS object describing the coordinate system to transform to
    output_shape
        pixel shape of the reprojected output array
        

    Returns
    -------
    np.ndarray
        output array after reprojection of the input array


    Example Call
    ------------

    output_array = reproject_array(input_array, input_wcs, output_wcs, output_shape)
    """

    output_array = reproject.reproject_adaptive((input_array, input_wcs), output_wcs, output_shape, roundtrip_coords=False, return_footprint=False)
    
    return output_array

# trefoil mosaic generation function
def mosaic(data_nfi1: np.ndarray,
            data_wfi1: np.ndarray,
            data_wfi2: np.ndarray,
            data_wfi3: np.ndarray,
            uncrt_nfi1: np.ndarray,
            uncrt_wfi1: np.ndarray,
            uncrt_wfi2: np.ndarray,
            uncrt_wfi3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    """PUNCH trefoil mosaic generation

    Taking a set of 3xWFI and 1xNFI observations, this function performs the 
        process by which they are meshed into a trefoil or mosaic observation.

    Parameters
    ----------
    data_nfi1
        reprojected data array from NFI 1
    data_wfi1
        reprojected data array from WFI 1
    data_wfi2
        reprojected data array from WFI 2
    data_wfi3
        reprojected data array from WFI 3
    uncrt_nfi1
        reprojected uncertainty array from NFI 1
    uncrt_wfi1
        reprojected uncertainty array from WFI 1
    uncrt_wfi2
        reprojected uncertainty array from WFI 2
    uncrt_wfi3
        reprojected uncertainty array from WFI 3

    Returns
    -------
    np.ndarray
        reprojected trefoil mosaic data array
    np.ndarray
        reprojected trefoil mosaic uncertainty array


    Example Call
    ------------

    (trefoil_data, trefoil_uncertainty) = mosaic(data_nfi1, data_wfi1, data_wfi2, data_wfi3, uncrt_nfi1, uncrt_wfi1, uncrt_wfi2, uncrt_wfi3)
    """
    
    temp_data = np.where(uncrt_wfi1 < uncrt_wfi2, data_wfi1, data_wfi2)
    temp_uncertainty = np.fmin(uncrt_wfi1, uncrt_wfi2)

    temp_data = np.where(temp_uncertainty < uncrt_wfi3, temp_data, data_wfi3)
    temp_uncertainty = np.fmin(temp_uncertainty, uncrt_wfi3)

    temp_data = np.where(temp_uncertainty < uncrt_nfi1, temp_data, data_nfi1)
    temp_uncertainty = np.fmin(temp_uncertainty, uncrt_nfi1)

    trefoil_data = temp_data
    trefoil_uncertainty = temp_uncertainty

    return (trefoil_data, trefoil_uncertainty)


# this is the core task associated with the module, it should end in "task" and
# use the @task decorator for prefect tasks.
# use the logger to track the prefect flow, and add history to the data object,
# an example from destreak is included below
@task
def module_task(data_object: PunchData) -> PunchData:
    logger = get_run_logger()
    logger.info("this module started")
    # do module stuff here in here
    data_array = data_object.data
    uncertainty_array = data_object.uncertainty
    data_wcs = data_object.wcs
    
    logger.info("this module finished")
    data_object.add_history(datetime.now(), "LEVEL1-module", "this module ran") 
    return data_object