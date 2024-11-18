
import numpy as np
import reproject
from astropy.wcs import WCS
from ndcube import NDCube
from prefect import flow
from sunpy.coordinates import sun

from punchbowl.data.wcs import calculate_celestial_wcs_from_helio
from punchbowl.prefect import punch_task


@punch_task
def reproject_cube(input_cube: NDCube, output_wcs: WCS, output_shape: tuple[int, int]) -> np.ndarray:
    """
    Core reprojection function.

    Core reprojection function of the PUNCH mosaic generation module.
        With an input data array and corresponding WCS object, the function
        performs a reprojection into the output WCS object system, along with
        a specified pixel size for the output array. This utilizes the adaptive
        reprojection routine implemented in the reprojection astropy package.

    Parameters
    ----------
    input_cube: NDCube
        input cube to be reprojected
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
    >>> reprojected_arrays = reproject_cube(input_cube, output_wcs, output_shape)

    """
    input_wcs = input_cube.wcs
    time = input_cube.meta.astropy_time

    celestial_source = calculate_celestial_wcs_from_helio(input_wcs, time, output_shape)
    celestial_target = calculate_celestial_wcs_from_helio(output_wcs, time, output_shape)

    celestial_source.wcs.set_pv([(2, 1, (-sun.earth_distance(time) / sun.constants.radius).decompose().value)])

    return reproject.reproject_adaptive(
        (np.stack([input_cube.data, input_cube.uncertainty.array]), celestial_source),
        celestial_target, output_shape,
        roundtrip_coords=False, return_footprint=False,
    )


@flow(validate_parameters=False)
def reproject_many_flow(data: list[NDCube | None], trefoil_wcs: WCS, trefoil_shape: np.ndarray) -> list[NDCube | None]:
    """Reproject many flow."""
    out_layers = [reproject_cube.submit(d, trefoil_wcs, trefoil_shape) if d is not None else None for d in data]

    return [NDCube(data=out_layers[i].result()[0],
                   uncertainty=out_layers[i].result()[1],
                   wcs=trefoil_wcs,
                   meta=d.meta) if d is not None else None for i, d in enumerate(data)]
