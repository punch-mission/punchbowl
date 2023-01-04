# Core Python imports
from datetime import datetime
from typing import Tuple, List

# Third party imports
import numpy as np
import reproject
from astropy.wcs import WCS
from prefect import task, flow, get_run_logger

# Punchbowl imports
from punchbowl.data import PUNCHData


# core reprojection function
@task
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

    output_array = reproject.reproject_adaptive((input_array, input_wcs), output_wcs,
                                                output_shape, roundtrip_coords=False, return_footprint=False)
    
    return output_array


# trefoil mosaic generation function
@task
def mosaic(data_input: List,
            uncertainty_input: List,
            wcs_input: List,
            wcs_output: WCS,
            shape_output: Tuple) -> Tuple[np.ndarray, np.ndarray]:

    """PUNCH trefoil mosaic generation

    Taking a set of 3xWFI and 1xNFI observations, this function performs the 
        process by which they are meshed into a trefoil or mosaic observation.

    Parameters
    ----------
    data_input
        list of ndarray data objects to assemble into a mosaic
    uncertainty_input
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

    (trefoil_data, trefoil_uncertainty) = mosaic(data_input, uncertainty_input, wcs_input,
                                                 wcs_output, shape_output)
    """
    
    reprojected_data = np.zeros([shape_output[0], shape_output[1], len(data_input)])
    reprojected_uncertainty = np.zeros([shape_output[0], shape_output[1], len(data_input)])

    i = 0
    for idata, iwcs in zip(data_input, wcs_input):
        reprojected_data[:,:,i] = reproject_array.fn(idata, iwcs, wcs_output, shape_output)
        i = i+1

    i = 0
    for iuncertainty, iwcs in zip(uncertainty_input, wcs_input):
        reprojected_uncertainty[:, :, i] = reproject_array.fn(iuncertainty, iwcs, wcs_output, shape_output)
        i = i+1

    # Merge these data
    # Carefully deal with missing data (NaN) by only ignoring a pixel missing from all observations
    trefoil_data = np.nansum(reprojected_data * reprojected_uncertainty, axis=2) / np.nansum(reprojected_uncertainty, axis=2)
    trefoil_uncertainty = np.amax(reprojected_uncertainty)

    return trefoil_data, trefoil_uncertainty


# core module flow
@flow
def quickpunch_merge_flow(data: List) -> PUNCHData:
    logger = get_run_logger()
    logger.info("QuickPUNCH_merge module started")

    # Define output WCS from file
    trefoil_wcs = WCS('data/trefoil_hdr.fits')
    trefoil_shape = trefoil_wcs.array_shape()

    # Unpack input data objects
    data_input, uncertainty_input, wcs_input = [], [], []
    for obj in data:
        data_input.append(obj.data)
        uncertainty_input.append(obj.uncertainty)
        wcs_input.append(obj.wcs)

    (trefoil_data, trefoil_uncertainty) = mosaic(data_input, uncertainty_input, wcs_input, trefoil_wcs, trefoil_shape)

    # Pack up an output data object
    data_object = PUNCHData(trefoil_data, uncertainty=trefoil_uncertainty, wcs=trefoil_wcs)
    
    logger.info("QuickPUNCH_merge flow finished")
    data_object.add_history(datetime.now(), "LEVEL1-module", "QuickPUNCH_merge ran")
    return data_object