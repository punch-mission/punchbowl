# Core Python imports
from typing import Optional, Tuple, List
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
def mosaic(data_input: List,
            uncert_input: List,
            wcs_input: List,
            wcs_output: WCS,
            shape_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    """PUNCH trefoil mosaic generation

    Taking a set of 3xWFI and 1xNFI observations, this function performs the 
        process by which they are meshed into a trefoil or mosaic observation.

    Parameters
    ----------
    data_input
        list of ndarray data objects to assemble into a mosaic
    uncert_input
        list of ndarray uncertainty objects to assemble into a mosaic
    wcs_input
        list of corresponding WCS objects utilized to assemble a mosaic
    wcs_output
        output wcs object for the trefoil mosaic
    shape_output
        specified output shape for reprojection
    

    Returns
    -------
    np.ndarray
        reprojected trefoil mosaic data array
    np.ndarray
        reprojected trefoil mosaic uncertainty array


    Example Call
    ------------

    (trefoil_data, trefoil_uncertainty) = mosaic(data_input, uncert_input, wcs_input, wcs_output, shape_output)
    """
    
    reprojected_data = np.zeros([shape_output[0], shape_output[1], len(data_input)])
    reprojected_uncert = np.zeros([shape_output[0], shape_output[1], len(data_input)])

    i = 0
    for idata, iwcs in zip(data_input, wcs_input):
        reprojected_data[:,:,i] = reproject_array(idata, iwcs, wcs_output, shape_output)
        i = i+1

    i = 0
    for iuncert, iwcs in zip(uncert_input, wcs_input):
        reprojected_uncert[:,:,i] = reproject_array(iuncert, iwcs, wcs_output, shape_output)
        i = i+1

    # Merge these data
    # TODO - carefully check how this deals with NaNs
    trefoil_data = ((reprojected_data * reprojected_uncert).sum(axis=2)) / (reprojected_uncert.sum(axis=2))
    trefoil_uncert = np.amax(reprojected_uncert)

    return (trefoil_data, trefoil_uncert)


# this is the core task associated with the module, it should end in "task" and
# use the @task decorator for prefect tasks.
# use the logger to track the prefect flow, and add history to the data object,
# an example from destreak is included below

# TODO - Think about flows...
# TODO - Refine meshing procedure with NaNs
# TODO - Rename variables to be consistant

@task
def QuickPUNCH_merge(data: List) -> PUNCHData:
    logger = get_run_logger()
    logger.info("this module started")

    # Define output WCS
    # TODO - should this be read from a template file, or passed in as an input?
    trefoil_shape = [4096,4096]

    trefoil_wcs = WCS(naxis=2)
    trefoil_wcs.wcs.crpix = trefoil_shape[1]/2, trefoil_shape[0]/2
    trefoil_wcs.wcs.crval = 0, 0
    trefoil_wcs.wcs.cdelt = 0.0225, 0.0225
    trefoil_wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"

    # Unpack input data objects
    data_input, uncert_input, wcs_input = []
    for obj in data:
        data_input.append(obj.data)
        uncert_input.append(obj.uncertainty)
        wcs_input.append(obj.wcs)

    (trefoil_data, trefoil_uncertainty) = mosaic(data_input, uncert_input, wcs_input, trefoil_wcs, trefoil_shape)

    # Pack up an output data object
    data_object = PUNCHData(trefoil_data, uncertainty=trefoil_uncertainty, wcs=trefoil_wcs)
    
    logger.info("this module finished")
    data_object.add_history(datetime.now(), "LEVEL1-module", "this module ran") 
    return data_object