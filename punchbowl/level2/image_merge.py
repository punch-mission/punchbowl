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
    >>> output_array = reproject_array(input_array, input_wcs, output_wcs, output_shape)
    """

    output_array = reproject.reproject_adaptive((input_array, input_wcs), output_wcs,
                                                output_shape, roundtrip_coords=False, return_footprint=False)
    
    return output_array



# core module flow
@flow
def image_merge_flow(data: List) -> PUNCHData:
    logger = get_run_logger()
    logger.info("image_merge module started")

    # Define output WCS from file
    trefoil_wcs = WCS('level2/data/trefoil_hdr.fits')
    trefoil_shape = (4096,4096)
    trefoil_wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"
    #trefoil_shape = trefoil_wcs.array_shape

    # Unpack input data objects
    data_input, uncertainty_input, wcs_input = [], [], []
    for obj in data:
        data_input.append(obj.data)
        uncertainty_input.append(obj.uncertainty)
        wcs_input.append(obj.wcs)

    reprojected_data = np.zeros([trefoil_shape[0], trefoil_shape[1], len(data_input)])
    reprojected_uncertainty = np.zeros([trefoil_shape[0], trefoil_shape[1], len(data_input)])

    data_result = [reproject_array.submit(idata, iwcs, trefoil_wcs, trefoil_shape)
                   for idata, iwcs in zip(data_input, wcs_input)]
    uncertainty_result = [reproject_array.submit(iuncertainty.array, iwcs, trefoil_wcs, trefoil_shape)
                          for iuncertainty, iwcs in zip(uncertainty_input, wcs_input)]

    for i, d in enumerate(data_result):
        reprojected_data[:, :, i] = d.result()

    for i, d in enumerate(uncertainty_result):
        reprojected_uncertainty[:, :, i] = d.result()

    # Merge these data
    # Carefully deal with missing data (NaN) by only ignoring a pixel missing from all observations
    trefoil_data = np.nansum(reprojected_data * reprojected_uncertainty, axis=2) / np.nansum(reprojected_uncertainty, axis=2)
    trefoil_uncertainty = np.amax(reprojected_uncertainty)

    # Pack up an output data object
    data_object = PUNCHData(trefoil_data, uncertainty=trefoil_uncertainty, wcs=trefoil_wcs)
    
    logger.info("image_merge flow finished")
    data_object.meta.history.add_now("LEVEL2-module", "image_merge ran")
    return data_object