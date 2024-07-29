from __future__ import annotations

import os
from datetime import datetime

import astropy.units as u
import astropy.wcs.wcsapi
import numpy as np
import sunpy.map
from astropy.coordinates import GCRS, EarthLocation, SkyCoord, StokesSymbol, custom_stokes_symbol_mapping
from astropy.wcs import WCS
from sunpy.coordinates import frames
from sunpy.coordinates.sun import _sun_north_angle_to_z

_ROOT = os.path.abspath(os.path.dirname(__file__))
PUNCH_STOKES_MAPPING = custom_stokes_symbol_mapping({10: StokesSymbol("pB", "polarized brightness"),
                                                     11: StokesSymbol("B", "total brightness")})


def extract_crota_from_wcs(wcs: WCS) -> tuple[float, float]:
    """Extract CROTA from a WCS."""
    return np.arctan2(wcs.wcs.pc[1, 0], wcs.wcs.pc[0, 0]) * u.rad


def calculate_helio_wcs_from_celestial(wcs_celestial: WCS, date_obs: datetime, data_shape: tuple[int]) -> WCS:
    """Calculate the helio WCS from a celestial WCS."""
    is_3d = len(data_shape) == 3

    # we're at the center of the Earth
    test_loc = EarthLocation.from_geocentric(0, 0, 0, unit=u.m)
    test_gcrs = SkyCoord(test_loc.get_gcrs(date_obs))

    # follow the SunPy tutorial from here
    # https://docs.sunpy.org/en/stable/generated/gallery/units_and_coordinates/radec_to_hpc_map.html#sphx-glr-generated-gallery-units-and-coordinates-radec-to-hpc-map-py
    reference_coord = SkyCoord(
        wcs_celestial.wcs.crval[0] * u.Unit(wcs_celestial.wcs.cunit[0]),
        wcs_celestial.wcs.crval[1] * u.Unit(wcs_celestial.wcs.cunit[1]),
        frame="gcrs",
        obstime=date_obs,
        obsgeoloc=test_gcrs.cartesian,
        obsgeovel=test_gcrs.velocity.to_cartesian(),
        distance=test_gcrs.hcrs.distance,
    )

    reference_coord_arcsec = reference_coord.transform_to(frames.Helioprojective(observer=test_gcrs))

    cdelt1 = (np.abs(wcs_celestial.wcs.cdelt[0]) * u.deg).to(u.arcsec)
    cdelt2 = (np.abs(wcs_celestial.wcs.cdelt[1]) * u.deg).to(u.arcsec)

    geocentric = GCRS(obstime=date_obs)
    p_angle = _sun_north_angle_to_z(geocentric)

    crota = extract_crota_from_wcs(wcs_celestial)

    new_header = sunpy.map.make_fitswcs_header(
        data_shape[1:] if is_3d else data_shape,
        reference_coord_arcsec,
        reference_pixel=u.Quantity(
            [wcs_celestial.wcs.crpix[0] - 1, wcs_celestial.wcs.crpix[1] - 1] * u.pixel,
        ),
        scale=u.Quantity([cdelt1, cdelt2] * u.arcsec / u.pix),
        rotation_angle=-p_angle - crota,
        observatory="PUNCH",
        projection_code=wcs_celestial.wcs.ctype[0][-3:],
    )

    wcs_helio = WCS(new_header)

    if is_3d:
        wcs_helio = astropy.wcs.utils.add_stokes_axis_to_wcs(wcs_helio, 2)

    return wcs_helio, p_angle


def load_trefoil_wcs() -> astropy.wcs.WCS:
    """Load Level 2 trefoil world coordinate system and shape."""
    trefoil_wcs = WCS(os.path.join(_ROOT, "data", "trefoil_hdr.fits"))
    trefoil_wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"  # TODO: figure out why this is necessary, seems like a bug
    trefoil_shape = (4096, 4096)
    return trefoil_wcs, trefoil_shape
