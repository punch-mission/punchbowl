import numpy as np
from astropy.nddata import StdDevUncertainty
from astropy.units import Unit
from astropy.wcs import WCS
from ndcube import NDCube

from punchbowl.data.meta import NormalizedMetadata


def add_celestial_wcs(cube: NDCube, celestial_wcs: WCS) -> NDCube:
    """Add a celestial WCS as an extra coordinate system on an NDCube."""
    cube.extra_coords.wcs = celestial_wcs
    cube.extra_coords.mapping = tuple(range(celestial_wcs.pixel_n_dim))
    return cube


def get_celestial_wcs(cube: NDCube) -> WCS | None:
    """Return the celestial WCS stored in an NDCube's extra coordinates."""
    return cube.extra_coords.wcs


def replace_cube(cube: NDCube, data: np.ndarray | None = None,
                 meta: NormalizedMetadata | None = None, wcs: WCS | None = None,
                 celestial_wcs: WCS | None = None, unit: Unit | None = None,
                 mask: np.ndarray | None = None,
                 uncertainty: StdDevUncertainty | None = None) -> NDCube:
    """Copy an NDCube, replacing the specified attributes."""
    output = NDCube(
        data=data if data is not None else cube.data,
        meta=meta if meta is not None else cube.meta,
        wcs=wcs if wcs is not None else cube.wcs,
        unit=unit if unit is not None else cube.unit,
        mask=mask if mask is not None else cube.mask,
        uncertainty=uncertainty if uncertainty is not None else cube.uncertainty,
    )
    output_celestial_wcs = celestial_wcs if celestial_wcs is not None else get_celestial_wcs(cube)
    if output_celestial_wcs is not None:
        add_celestial_wcs(output, output_celestial_wcs)
    return output
