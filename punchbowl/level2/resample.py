from datetime import datetime

import numpy as np
import reproject
from astropy.wcs import WCS
from ndcube import NDCube
from prefect import flow, task
from sunpy.coordinates import sun

from punchbowl.data.wcs import calculate_celestial_wcs_from_helio


@task
def reproject_array(input_array: np.ndarray, input_wcs: WCS, time: datetime,
                    output_wcs: WCS, output_shape: tuple) -> np.ndarray:
    """
    Core reprojection function.

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
    time
        time of the observation to reproject
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
    celestial_source = calculate_celestial_wcs_from_helio(input_wcs, time, output_shape)
    celestial_target = calculate_celestial_wcs_from_helio(output_wcs, time, output_shape)

    celestial_source.wcs.set_pv([(2, 1, (-sun.earth_distance(time) / sun.constants.radius).decompose().value)])

    return reproject.reproject_adaptive(
        (input_array, celestial_source), celestial_target, output_shape,
        roundtrip_coords=False, return_footprint=False,
    )


@flow(validate_parameters=False)
def reproject_many_flow(data: list[NDCube], trefoil_wcs: WCS, trefoil_shape: np.ndarray) -> list[NDCube]:
    """Reproject many flow."""
    data_result = [reproject_array.submit(d.data, d.wcs, d.meta.astropy_time, trefoil_wcs, trefoil_shape) for d in data]
    uncertainty_result = [reproject_array.submit(d.uncertainty.array, d.wcs,
                                                 d.meta.astropy_time, trefoil_wcs, trefoil_shape) for d in data]

    return [NDCube(data=data_result[i].result(),
                   uncertainty=uncertainty_result[i].result(),
                   wcs=trefoil_wcs,
                   meta=d.meta) for i, d in enumerate(data)]
