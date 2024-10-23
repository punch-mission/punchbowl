from __future__ import annotations

import os
from datetime import datetime

import astropy.coordinates
import astropy.units as u
import astropy.wcs.wcsapi
import numpy as np
import sunpy.coordinates
import sunpy.map
from astropy.coordinates import GCRS, EarthLocation, SkyCoord, StokesSymbol, custom_stokes_symbol_mapping, get_sun
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import add_stokes_axis_to_wcs
from sunpy.coordinates import frames

_ROOT = os.path.abspath(os.path.dirname(__file__))
PUNCH_STOKES_MAPPING = custom_stokes_symbol_mapping({10: StokesSymbol("pB", "polarized brightness"),
                                                     11: StokesSymbol("B", "total brightness")})


def extract_crota_from_wcs(wcs: WCS) -> tuple[float, float]:
    """Extract CROTA from a WCS."""
    delta_ratio = wcs.wcs.cdelt[1] / wcs.wcs.cdelt[0]
    return np.arctan2(wcs.wcs.pc[1, 0]/delta_ratio, wcs.wcs.pc[0, 0]) * u.rad


def calculate_helio_wcs_from_celestial(wcs_celestial: WCS,
                                       date_obs: datetime,
                                       data_shape: tuple[int, int]) -> (WCS, float):
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

    p_angle = get_p_angle(date_obs)

    rotation_angle = extract_crota_from_wcs(wcs_celestial).to(u.rad).value - get_p_angle(date_obs).rad
    new_pc_matrix = calculate_pc_matrix(rotation_angle, wcs_celestial.wcs.cdelt)

    projection_code = wcs_celestial.wcs.ctype[0][-3:] if "-" in wcs_celestial.wcs.ctype[0] else ""
    if projection_code:  # noqa: SIM108
        new_ctypes = (f"HPLN-{projection_code}", f"HPLT-{projection_code}")
    else:
        new_ctypes = "HPLN", "HPLT"

    wcs_helio = WCS(naxis=2)
    wcs_helio.wcs.ctype = new_ctypes
    wcs_helio.wcs.cunit = ("deg", "deg")
    wcs_helio.wcs.cdelt = (cdelt1.to(u.deg).value, cdelt2.to(u.deg).value)
    wcs_helio.wcs.crpix = (wcs_celestial.wcs.crpix[0], wcs_celestial.wcs.crpix[1])
    wcs_helio.wcs.crval = (reference_coord_arcsec.Tx.to(u.deg).value,  reference_coord_arcsec.Ty.to(u.deg).value)
    wcs_helio.wcs.pc = new_pc_matrix
    wcs_helio.wcs.set_pv(wcs_celestial.wcs.get_pv())

    if is_3d:
        wcs_helio = astropy.wcs.utils.add_stokes_axis_to_wcs(wcs_helio, 2)

    return wcs_helio, p_angle


def get_sun_ra_dec(dt: datetime) -> (float, float):
    """Get the position of the Sun in right ascension and declination."""
    position = get_sun(Time(str(dt), scale="utc"))
    return position.ra.value, position.dec.value


def calculate_pc_matrix(crota: float, cdelt: (float, float)) -> np.ndarray:
    """
    Calculate a PC matrix given CROTA and CDELT.

    Parameters
    ----------
    crota : float
        rotation angle from the WCS
    cdelt : float
        pixel size from the WCS

    Returns
    -------
    np.ndarray
        PC matrix

    """
    return np.array(
        [
            [np.cos(crota), -np.sin(crota) * (cdelt[0] / cdelt[1])],
            [np.sin(crota) * (cdelt[1] / cdelt[0]), np.cos(crota)],
        ],
    )


def _times_are_equal(time_1: astropy.time.Time, time_2: astropy.time.Time) -> bool:
    # Ripped from sunpy, modified
    # Checks whether times are equal
    if isinstance(time_1, astropy.time.Time) and isinstance(time_2, astropy.time.Time):
        # We explicitly perform the check in TAI to avoid possible numerical precision differences
        # between a time in UTC and the same time after a UTC->TAI->UTC conversion
        return np.all(time_1.tai == time_2.tai)

    # We also deem the times equal if one is None
    return time_1 is None or time_2 is None


def get_p_angle(time: str="now") -> u.deg:
    """
    Get the P angle.

    Return the position (P) angle for the Sun at a specified time, which is the angle between
    geocentric north and solar north as seen from Earth, measured eastward from geocentric north.
    The range of P is +/-26.3 degrees.

    Parameters
    ----------
    time : {parse_time_types}
        Time to use in a parse_time-compatible format

    Returns
    -------
    out : `~astropy.coordinates.Angle`
        The position angle

    """
    obstime = sunpy.coordinates.sun.parse_time(time)

    # Define the frame where its Z axis is aligned with geocentric north
    geocentric = astropy.coordinates.GCRS(obstime=obstime)

    return sunpy.coordinates.sun._sun_north_angle_to_z(geocentric)  # noqa: SLF001


@astropy.coordinates.frame_transform_graph.transform(
    astropy.coordinates.FunctionTransform,
    sunpy.coordinates.Helioprojective,
    astropy.coordinates.GCRS)
def hpc_to_gcrs(HPcoord, GCRSframe):  # noqa: ANN201, N803, ANN001
    """Convert helioprojective to GCRS."""
    if not _times_are_equal(HPcoord.obstime, GCRSframe.obstime):
        raise ValueError("Obstimes are not equal")  # noqa: TRY003, EM101
    obstime = HPcoord.obstime or GCRSframe.obstime

    # Compute the three angles we need
    position = astropy.coordinates.get_sun(obstime)
    ra, dec = position.ra.rad, position.dec.rad
    p = get_p_angle(obstime).rad

    # Prepare rotation matrices for each
    p_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(p), -np.sin(p)],
        [0, np.sin(p), np.cos(p)],
    ])

    ra_matrix = np.array([
        [np.cos(ra), -np.sin(ra), 0],
        [np.sin(ra), np.cos(ra), 0],
        [0, 0, 1],
    ])

    dec_matrix = np.array([
        [np.cos(-dec), 0, np.sin(-dec)],
        [0, 1, 0],
        [-np.sin(-dec), 0, np.cos(-dec)],
    ])

    # Compose the matrices
    matrix = ra_matrix @ dec_matrix @ p_matrix

    # Extract the input coordinates and negate the HP latitude,
    # since it increases in the opposite direction from RA.
    if HPcoord._is_2d:  # noqa: SLF001
        rep = astropy.coordinates.UnitSphericalRepresentation(
            -HPcoord.Tx, HPcoord.Ty)
    else:
        rep = astropy.coordinates.SphericalRepresentation(
            -HPcoord.Tx, HPcoord.Ty, HPcoord.distance)

    # Apply the transformation
    rep = rep.to_cartesian()
    rep = rep.transform(matrix)

    # Match the input representation. (If the input was UnitSpherical, meaning there's no
    # distance coordinate, this drops the distance coordinate.)
    rep = rep.represent_as(type(HPcoord.data))

    # Put the computed coordinates into the output frame
    return GCRSframe.realize_frame(rep)


@astropy.coordinates.frame_transform_graph.transform(
    astropy.coordinates.FunctionTransform,
    astropy.coordinates.GCRS,
    sunpy.coordinates.Helioprojective)
def gcrs_to_hpc(GCRScoord, Helioprojective): # noqa: ANN201, N803, ANN001
    """Convert GCRS to HPC."""
    if not _times_are_equal(GCRScoord.obstime, Helioprojective.obstime):
        raise ValueError("Obstimes are not equal")  # noqa: TRY003, EM101
    obstime = GCRScoord.obstime or Helioprojective.obstime

    # Compute the three angles we need
    position = astropy.coordinates.get_sun(obstime)
    ra, dec = position.ra.rad, position.dec.rad
    p = get_p_angle(obstime).rad

    # Prepare rotation matrices for each
    p_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(p), -np.sin(p)],
        [0, np.sin(p), np.cos(p)],
    ])

    ra_matrix = np.array([
        [np.cos(ra), -np.sin(ra), 0],
        [np.sin(ra), np.cos(ra), 0],
        [0, 0, 1],
    ])

    dec_matrix = np.array([
        [np.cos(-dec), 0, np.sin(-dec)],
        [0, 1, 0],
        [-np.sin(-dec), 0, np.cos(-dec)],
    ])

    # Compose the matrices
    old_matrix = ra_matrix @ dec_matrix @ p_matrix
    matrix = np.linalg.inv(old_matrix)

    # Extract the input coordinates and negate the HP latitude,
    # since it increases in the opposite direction from RA.
    if isinstance(GCRScoord.data, astropy.coordinates.UnitSphericalRepresentation):
        rep = astropy.coordinates.UnitSphericalRepresentation(
            GCRScoord.ra, GCRScoord.dec)  # , earth_distance(obstime))
    else:
        rep = astropy.coordinates.SphericalRepresentation(
            GCRScoord.ra, GCRScoord.dec, GCRScoord.distance)

    # Apply the transformation
    rep = rep.to_cartesian()
    rep = rep.transform(matrix)

    # Match the input representation. (If the input was UnitSpherical, meaning there's no
    # distance coordinate, this drops the distance coordinate.)
    rep = rep.represent_as(type(GCRScoord.data))

    if isinstance(rep, astropy.coordinates.UnitSphericalRepresentation):
        rep = astropy.coordinates.UnitSphericalRepresentation(
            -rep.lon, rep.lat)  # , earth_distance(obstime))
    else:
        rep = astropy.coordinates.SphericalRepresentation(
            -rep.lon, rep.lat, rep.distance)

    # Put the computed coordinates into the output frame
    return Helioprojective.realize_frame(rep)


def calculate_celestial_wcs_from_helio(wcs_helio: WCS, date_obs: datetime, data_shape: tuple[int, int]) -> WCS:
    """Calculate the celestial WCS from a helio WCS."""
    is_3d = len(data_shape) == 3

    old_crval = SkyCoord(wcs_helio.wcs.crval[0] * u.deg, wcs_helio.wcs.crval[1] * u.deg,
                         frame="helioprojective", observer="earth", obstime=date_obs)
    new_crval = old_crval.transform_to(GCRS)

    rotation_angle = extract_crota_from_wcs(wcs_helio).to(u.rad).value - get_p_angle(date_obs).rad
    new_pc_matrix = calculate_pc_matrix(rotation_angle, wcs_helio.wcs.cdelt)

    cdelt1 = np.abs(wcs_helio.wcs.cdelt[0]) * u.deg
    cdelt2 = np.abs(wcs_helio.wcs.cdelt[1]) * u.deg

    projection_code = wcs_helio.wcs.ctype[0][-3:] if "-" in wcs_helio.wcs.ctype[0] else ""
    if projection_code:  # noqa: SIM108
        new_ctypes = (f"RA---{projection_code}", f"DEC--{projection_code}")
    else:
        new_ctypes = "RA", "DEC"


    wcs_celestial = WCS(naxis=2)
    wcs_celestial.wcs.ctype = new_ctypes
    wcs_celestial.wcs.cunit = ("deg", "deg")
    wcs_celestial.wcs.cdelt = (-cdelt1.to(u.deg).value, cdelt2.to(u.deg).value)
    wcs_celestial.wcs.crpix = (wcs_helio.wcs.crpix[0], wcs_helio.wcs.crpix[1])
    wcs_celestial.wcs.crval = (new_crval.ra.to(u.deg).value,  new_crval.dec.to(u.deg).value)
    wcs_celestial.wcs.pc = new_pc_matrix
    wcs_celestial.wcs.set_pv(wcs_helio.wcs.get_pv())

    if is_3d:
        wcs_celestial = add_stokes_axis_to_wcs(wcs_celestial, 2)

    return wcs_celestial


def load_trefoil_wcs() -> (astropy.wcs.WCS, (int, int)):
    """Load Level 2 trefoil world coordinate system and shape."""
    trefoil_wcs = WCS(os.path.join(_ROOT, "data", "trefoil_wcs.fits"))
    trefoil_wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"  # TODO: figure out why this is necessary, seems like a bug
    trefoil_shape = (4096, 4096)
    return trefoil_wcs, trefoil_shape


def load_quickpunch_mosaic_wcs() -> (astropy.wcs.WCS, (int, int)):
    """Load Level quickPUNCH mosaic world coordinate system and shape."""
    quickpunch_mosaic_shape = (1024, 1024)
    quickpunch_mosaic_wcs = WCS(naxis=2)

    quickpunch_mosaic_wcs.wcs.crpix = quickpunch_mosaic_shape[1] / 2 + 0.5, quickpunch_mosaic_shape[0] / 2 + 0.5
    quickpunch_mosaic_wcs.wcs.crval = 0, 0
    quickpunch_mosaic_wcs.wcs.cdelt = 0.045, 0.045
    quickpunch_mosaic_wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"

    return quickpunch_mosaic_wcs, quickpunch_mosaic_shape


def load_quickpunch_nfi_wcs() -> (astropy.wcs.WCS, (int, int)):
    """Load Level quickPUNCH NFI world coordinate system and shape."""
    quickpunch_nfi_shape = (1024, 1024)
    quickpunch_nfi_wcs = WCS(naxis=2)

    quickpunch_nfi_wcs.wcs.crpix = quickpunch_nfi_shape[1] / 2 + 0.5, quickpunch_nfi_shape[0] / 2 + 0.5
    quickpunch_nfi_wcs.wcs.crval = 0, 0
    quickpunch_nfi_wcs.wcs.cdelt = 30 / 3600 * 2, 30 / 3600 * 2
    quickpunch_nfi_wcs.wcs.ctype = "HPLN-TAN", "HPLT-TAN"

    return quickpunch_nfi_wcs, quickpunch_nfi_shape
