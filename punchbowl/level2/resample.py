# Core Python imports
from typing import List

# Third party imports
import numpy as np
import reproject
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from prefect import flow, task

# Punchbowl imports
from punchbowl.data import PUNCHData


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
    >>> reprojected_array = reproject_array(input_array, input_wcs, output_wcs, output_shape)
    """
    reconstructed_wcs = WCS(naxis=2)
    reconstructed_wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"
    reconstructed_wcs.wcs.cunit = "deg", "deg"
    reconstructed_wcs.wcs.cdelt = input_wcs.wcs.cdelt
    reconstructed_wcs.wcs.crpix = input_wcs.wcs.crpix
    reconstructed_wcs.wcs.crval = input_wcs.wcs.crval
    reconstructed_wcs.wcs.cname = "HPC lon", "HPC lat"

    return reproject.reproject_adaptive((input_array, reconstructed_wcs),
                                        output_wcs,
                                        output_shape,
                                        roundtrip_coords=False,
                                        return_footprint=False)


@flow(validate_parameters=False)
def reproject_many_flow(data: List[PUNCHData], trefoil_wcs: WCS, trefoil_shape: np.ndarray) -> List[PUNCHData]:
    # TODO: add docstring
    data_result = [reproject_array.submit(d.data, d.wcs, trefoil_wcs, trefoil_shape)
                   for d in data]
    uncertainty_result = [reproject_array.submit(d.uncertainty.array, d.wcs, trefoil_wcs, trefoil_shape)
                          for d in data]

    return [d.duplicate_with_updates(data=data_result[i].result(),
                                     uncertainty=StdDevUncertainty(uncertainty_result[i].result()),
                                     wcs=trefoil_wcs) for i, d in enumerate(data)]
