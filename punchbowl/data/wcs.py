from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING
from datetime import datetime

import astropy.coordinates
import astropy.time
import astropy.units as u
import astropy.wcs.wcsapi
import numpy as np
import sunpy.coordinates
import sunpy.map
from astropy.coordinates import (
    GCRS,
    ICRS,
    CartesianDifferential,
    EarthLocation,
    SkyCoord,
    StokesSymbol,
    custom_stokes_symbol_mapping,
    get_sun,
)
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning, WCSBase
from astropy.wcs.utils import add_stokes_axis_to_wcs
from sunpy.coordinates import frames

if TYPE_CHECKING:
    from punchbowl.data import NormalizedMetadata

_ROOT = os.path.abspath(os.path.dirname(__file__))
PUNCH_STOKES_MAPPING = custom_stokes_symbol_mapping({10: StokesSymbol("pB", "polarized brightness"),
                                                     11: StokesSymbol("B", "total brightness")})


def extract_crota_from_wcs(wcs: WCS) -> u.deg:
    """Extract CROTA from a WCS."""
    delta_ratio = wcs.wcs.cdelt[1] / wcs.wcs.cdelt[0]
    return np.arctan2(wcs.wcs.pc[1, 0]/delta_ratio, wcs.wcs.pc[0, 0]) * u.rad


def calculate_helio_wcs_from_celestial(wcs_celestial: WCS,
                                       date_obs: astropy.time.Time | str | None=None,
                                       data_shape: tuple[int, int] | None=None) -> WCS:
    """Calculate the helio WCS from a celestial WCS."""
    if not date_obs:
        date_obs = wcs_celestial.wcs.dateobs
    if data_shape is None:
        data_shape = wcs_celestial.array_shape
    if not date_obs:
        msg = "dateobs must be provided, either as an argument or inside the WCS"
        raise ValueError(msg)
    date_obs = Time(date_obs)
    is_3d = len(data_shape) == 3

    # we're at the center of the Earth
    test_loc = EarthLocation.from_geocentric(0, 0, 0, unit=u.m)
    test_gcrs = SkyCoord(test_loc.get_gcrs(date_obs))

    # follow the SunPy tutorial from here
    # https://docs.sunpy.org/en/stable/generated/gallery/units_and_coordinates/radec_to_hpc_map.html#sphx-glr-generated-gallery-units-and-coordinates-radec-to-hpc-map-py
    reference_coord = SkyCoord(
        wcs_celestial.wcs.crval[0] * u.Unit(wcs_celestial.wcs.cunit[0]),
        wcs_celestial.wcs.crval[1] * u.Unit(wcs_celestial.wcs.cunit[1]),
        9e99 * u.lyr,
        frame="gcrs",
        obstime=date_obs,
        obsgeoloc=test_gcrs.cartesian,
        obsgeovel=test_gcrs.velocity.to_cartesian(),
    )

    reference_coord_hp = reference_coord.transform_to(frames.Helioprojective(observer=test_gcrs))
    projection_code = wcs_celestial.wcs.ctype[0][-3:] if "-" in wcs_celestial.wcs.ctype[0] else ""

    hdr = sunpy.map.make_fitswcs_header(
        np.empty(data_shape),
        reference_coord_hp,
        reference_pixel=wcs_celestial.wcs.crpix * u.pix - 1 * u.pix,
        scale=np.abs(wcs_celestial.wcs.cdelt) * u.deg / u.pix,
        projection_code=projection_code)

    wcs_helio = WCS(hdr, naxis=2)
    wcs_helio.wcs.set_pv(wcs_celestial.wcs.get_pv())

    rotation_angle = -compute_hp_to_eq_rotation_angle(wcs_helio, date_obs)
    rotation_angle -= extract_crota_from_wcs(wcs_celestial)
    new_pc_matrix = calculate_pc_matrix(rotation_angle, wcs_helio.wcs.cdelt)
    wcs_helio.wcs.pc = new_pc_matrix
    wcs_helio.cpdis1 = wcs_celestial.cpdis1
    wcs_helio.cpdis2 = wcs_celestial.cpdis2

    if is_3d:
        wcs_helio = astropy.wcs.utils.add_stokes_axis_to_wcs(wcs_helio, 2)
    wcs_helio.array_shape = data_shape

    return wcs_helio


def get_sun_ra_dec(dt: datetime) -> tuple[float, float]:
    """Get the position of the Sun in right ascension and declination."""
    position = get_sun(Time(str(dt), scale="utc"))
    return position.ra.value, position.dec.value


def calculate_pc_matrix(crota: float, cdelt: tuple[float, float]) -> np.ndarray:
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


def hpc_to_gcrs(hp_coord):  # noqa: ANN201, ANN001
    """Convert helioprojective to GCRS."""
    if isinstance(hp_coord, SkyCoord):
        hp_coord = hp_coord.frame
    obstime = hp_coord.obstime

    # Compute the three angles we need
    position = astropy.coordinates.get_sun(obstime)
    ra, dec = position.ra.rad, position.dec.rad
    p = -get_p_angle(obstime).rad

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
    if hp_coord._is_2d:  # noqa: SLF001
        rep = astropy.coordinates.UnitSphericalRepresentation(
            -hp_coord.Tx, hp_coord.Ty)
    else:
        rep = astropy.coordinates.SphericalRepresentation(
            -hp_coord.Tx, hp_coord.Ty, hp_coord.distance)

    # Apply the transformation
    rep = rep.to_cartesian()
    rep = rep.transform(matrix)

    # Match the input representation. (If the input was UnitSpherical, meaning there's no
    # distance coordinate, this drops the distance coordinate.)
    rep = rep.represent_as(type(hp_coord.data))

    # Put the computed coordinates into the output frame
    return GCRS(obstime=obstime).realize_frame(rep)


def calculate_celestial_wcs_from_helio(wcs_helio: WCS,
                                       date_obs: astropy.time.Time | str | None=None,
                                       data_shape: tuple[int, int] | None=None) -> WCS:
    """Calculate the celestial WCS from a helio WCS."""
    if not date_obs:
        date_obs = wcs_helio.wcs.dateobs
    if data_shape is None:
        data_shape = wcs_helio.array_shape
    if not date_obs:
        msg = "dateobs must be provided, either as an argument or inside the WCS"
        raise ValueError(msg)
    is_3d = len(data_shape) == 3

    old_crval = SkyCoord(wcs_helio.wcs.crval[0] * u.deg, wcs_helio.wcs.crval[1] * u.deg,
                         9e99*u.lyr,
                         frame="helioprojective", observer="earth", obstime=date_obs)
    new_crval = old_crval.transform_to(GCRS)

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
    wcs_celestial.wcs.set_pv(wcs_helio.wcs.get_pv())
    wcs_celestial.wcs.dateobs = wcs_helio.wcs.dateobs
    wcs_celestial.wcs.mjdobs = wcs_helio.wcs.mjdobs
    wcs_celestial.cpdis1 = wcs_helio.cpdis1
    wcs_celestial.cpdis2 = wcs_helio.cpdis2

    rotation_angle = compute_hp_to_eq_rotation_angle(wcs_helio, date_obs)
    rotation_angle += extract_crota_from_wcs(wcs_helio)
    new_pc_matrix = calculate_pc_matrix(rotation_angle, wcs_helio.wcs.cdelt)
    wcs_celestial.wcs.pc = new_pc_matrix

    if is_3d:
        wcs_celestial = add_stokes_axis_to_wcs(wcs_celestial, 2)
    wcs_celestial.array_shape = data_shape

    return wcs_celestial


def project_vec_onto_plane(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Project vec1 onto the plane perpendicular to vec2."""
    # First project vec1 onto vec2
    v1_projected = np.dot(vec1, vec2) * vec2 / np.dot(vec2, vec2)
    # Now subtract that from vec1 to get the component perpendicular to vec2
    return vec1 - v1_projected


def angle_between_vectors(vec1: np.ndarray, vec2: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """
    Compute the angle between vec1 and vec2.

    The angle is computed in the plane normal to plane_normal. This normal vector also sets the handedness and the sign
    of the angle.
    """
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    return np.arctan2(np.dot(np.cross(vec1, vec2), plane_normal), np.dot(vec1, vec2))


def compute_hp_to_eq_rotation_angle(wcs_helio: WCS, date_obs: str | Time | None=None) -> u.Quantity:
    """
    Compute the necessary rotation angle to align a celestial WCS to a helioprojective one.

    The computed angle includes the effect of the P angle but not the rotation of the helioprojective WCS

    Parameters
    ----------
    wcs_helio : WCS
        A helioprojective WCS for which we're producing a corresponding celestial WCS. Only CRVAL is used from this WCS
    date_obs : str | Time | None
        The observation date. If not provided, it will be pulled from wcs_helio. If not provided in wcs_helio, it must
        be specified here.

    Returns
    -------
    rotation_angle : Quantity
        The rotation angle that should be added to that in wcs_celestial's PC matrix

    """
    if not date_obs:
        date_obs = wcs_helio.wcs.dateobs
    if not date_obs:
        msg = "dateobs must be provided, either as an argument or inside the WCS"
        raise ValueError(msg)
    # Get our CRVAL vector
    axis_sc = wcs_helio.celestial.pixel_to_world(wcs_helio.wcs.crpix[0] - 1, wcs_helio.wcs.crpix[1] - 1)
    # Make sure date_obs and observer are specified
    axis_sc = SkyCoord(axis_sc.data, frame="helioprojective", observer=axis_sc.observer or "earth", obstime=date_obs)
    axis = axis_sc.cartesian.xyz
    axis /= np.linalg.norm(axis)

    # Create a new vector pointing near north. Projected into the plane perpendicular to axis, this will point as
    # north as possible in that plane.
    northward = axis.copy()
    northward[2] += 20
    northward /= np.linalg.norm(northward)

    # Package up that vector and convert to our target coordinate system
    northward = SkyCoord(*northward, representation_type="cartesian", frame=axis_sc.frame)
    northward.representation_type = "spherical"
    rotated = hpc_to_gcrs(northward)
    rotated = rotated.cartesian.xyz
    rotated /= np.linalg.norm(rotated)

    # And also transform the CRVAL axis
    axis_rotated = hpc_to_gcrs(axis_sc)
    axis_rotated = axis_rotated.cartesian.xyz
    axis_rotated /= np.linalg.norm(axis_rotated)

    # Create a new northward vector in the target frame
    northward_in_new_frame = axis_rotated.copy()
    northward_in_new_frame[2] += 20
    northward_in_new_frame /= np.linalg.norm(northward_in_new_frame)

    # Project the old and new northward vectors into the plane perpendicular to axis
    northward_in_new_frame_proj = project_vec_onto_plane(northward_in_new_frame, axis_rotated)
    rotated_proj = project_vec_onto_plane(rotated, axis_rotated)

    return -angle_between_vectors(northward_in_new_frame_proj, rotated_proj, axis_rotated).to(u.deg)


def load_trefoil_wcs() -> tuple[astropy.wcs.WCS, tuple[int, int]]:
    """Load Level 2 trefoil world coordinate system and shape."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*The WCS transformation has more axes \\(2\\) than the image it is "
                                          "associated with \\(0\\).*", FITSFixedWarning)
        trefoil_wcs = WCS(os.path.join(_ROOT, "data", "trefoil_wcs.fits"))
    trefoil_wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"  # TODO: figure out why this is necessary, seems like a bug
    trefoil_shape = (4096, 4096)

    trefoil_wcs.array_shape = trefoil_shape

    return trefoil_wcs, trefoil_shape


def load_quickpunch_mosaic_wcs() -> tuple[astropy.wcs.WCS, tuple[int, int]]:
    """Load Level quickPUNCH mosaic world coordinate system and shape."""
    quickpunch_mosaic_shape = (1024, 1024)
    quickpunch_mosaic_wcs = WCS(naxis=2)

    quickpunch_mosaic_wcs.wcs.crpix = quickpunch_mosaic_shape[1] / 2 + 0.5, quickpunch_mosaic_shape[0] / 2 + 0.5
    quickpunch_mosaic_wcs.wcs.crval = 0, 0
    quickpunch_mosaic_wcs.wcs.cdelt = 0.045, 0.045
    quickpunch_mosaic_wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"

    quickpunch_mosaic_wcs.array_shape = quickpunch_mosaic_shape

    return quickpunch_mosaic_wcs, quickpunch_mosaic_shape


def load_quickpunch_nfi_wcs() -> tuple[astropy.wcs.WCS, tuple[int, int]]:
    """Load Level quickPUNCH NFI world coordinate system and shape."""
    quickpunch_nfi_shape = (1024, 1024)
    quickpunch_nfi_wcs = WCS(naxis=2)

    quickpunch_nfi_wcs.wcs.crpix = quickpunch_nfi_shape[1] / 2 + 0.5, quickpunch_nfi_shape[0] / 2 + 0.5
    quickpunch_nfi_wcs.wcs.crval = 0, 0
    quickpunch_nfi_wcs.wcs.cdelt = 30 / 3600 * 2, 30 / 3600 * 2
    quickpunch_nfi_wcs.wcs.ctype = "HPLN-TAN", "HPLT-TAN"

    quickpunch_nfi_wcs.array_shape = quickpunch_nfi_shape

    return quickpunch_nfi_wcs, quickpunch_nfi_shape


def celestial_north_from_wcs(input_wcs: WCS,
                             shape: tuple,
                             eps: u.Quantity=1.0*u.arcsec) -> np.ndarray:
    """
    Compute the angle of celestial north direction at each pixel using the WCS.

    Parameters
    ----------
    input_wcs : astropy.wcs.WCS
        2D celestial WCS (e.g., RA/Dec TAN) or a WCS where the celestial axes
        are the first two. If WCS has extra axes (e.g., STOKES), pass a
        dropped/sliced 2D WCS (e.g., wcs.dropaxis(2)).

    shape : tuple
        Shape of the image as (nrows, ncols).

    eps : astropy.units.Quantity
        Small sky offset applied toward celestial north (positive Dec direction)
        to estimate the local north direction.

    Returns
    -------
    angle_cel_north : 2D numpy.ndarray
        Angle in degrees from +Y image axis to celestial north at each pixel
        (measured counterclockwise in the pixel coordinate system):
            angle = atan2(dx, dy)

    """
    nrows, ncols = shape
    y, x = np.mgrid[0:nrows, 0:ncols]
    c = input_wcs.pixel_to_world(x, y)

    # Move a tiny amount toward celestial north (increasing Dec) at constant RA
    # spherical_offsets_by(dlon, dlat): dlon eastward, dlat northward
    cn = c.spherical_offsets_by(0*u.deg, eps)

    # World -> pixel for the north-offset point
    xn, yn = input_wcs.world_to_pixel(cn)

    # Pixel-space "north" direction vectors
    dx = xn - x
    dy = yn - y

    # Normalize vectors to unit length (avoid divide-by-zero)
    norm = np.hypot(dx, dy)
    north_dx = np.zeros_like(dx, dtype=np.float64)
    north_dy = np.zeros_like(dy, dtype=np.float64)
    good = norm > 0
    north_dx[good] = dx[good] / norm[good]
    north_dy[good] = dy[good] / norm[good]

    # Angle from +Y axis to north vector, -ve for CCW: atan2(dx, dy)
    angle_cel_north = -np.degrees(np.arctan2(north_dx, north_dy))

    return angle_cel_north*u.degree


class GCRSWCS(WCS):
    """
    A WCS that accepts and returns GCRS SkyCoords.

    This is necessary because there doesn't seem to be a native way to tell a WCS it is GCRS. I think we could register
    a definition with Astropy that if the FITS RADESYS key is set to 'gcrs' it should use GCRS SkyCoords, but the
    velocity information in the header doesn't appear to be read into the WCS object, and at the point where we're
    registering this with Astropy, we only have access to the WCS object, not the original header.
    """

    def __init__(self, *args: tuple, meta: NormalizedMetadata, **kwargs: dict) -> None:
        """
        Initialize the GCRSWCS.

        Reqquires the NormalizedMetadata to ready out the spacecraft position and velocity.
        """
        super().__init__(*args, **kwargs)

        self._meta = meta
        for ax in "XYZ":
            if meta.get(f"GEO{ax}_VOB", None) is None:
                self._gcrs_frame = ICRS()
                return
        for part in ("LON", "LAT", "ALT"):
            if meta.get(f"GEOD_{part}", None) is None:
                self._gcrs_frame = ICRS()
                return

        position = EarthLocation.from_geodetic(meta["GEOD_LON"].value * u.deg,
                                               meta["GEOD_LAT"].value * u.deg,
                                               meta["GEOD_ALT"].value * u.m)

        velocity = CartesianDifferential(
            meta["GEOX_VOB"].value * u.m / u.s,
            meta["GEOY_VOB"].value * u.m / u.s,
            meta["GEOZ_VOB"].value * u.m / u.s)

        # Re-create with velocity attached
        itrs = position.get_itrs(meta.astropy_time)
        newdata = itrs.data.to_cartesian().with_differentials(velocity)
        itrs = itrs.realize_frame(newdata)

        sc_gcrs = itrs.transform_to(GCRS(obstime=meta.astropy_time))

        self._gcrs_frame = GCRS(obsgeoloc=sc_gcrs.cartesian.without_differentials(),
                                obsgeovel=sc_gcrs.cartesian.differentials["s"].d_xyz,
                                obstime=meta.astropy_time)

    def _get_components_and_classes(self) -> tuple:
        """
        Determine the coordinate frame for a SkyCoord.

        When creating a SkyCoord, this is the method WCS objects use to select a coordinate frame. Rather than
        registering something within the default implementation, we override and replace the coordinate frame with
        our GCRS frame.
        """
        rval = super()._get_components_and_classes()
        rval[1]["celestial"][2]["frame"] = self._gcrs_frame
        return rval

    def __copy__(self) -> GCRSWCS:
        """
        Return a shallow copy of this WCS.

        We need to override this to pass the NormalizedMeta to the new object's constructor.
        """
        new_copy = self.__class__(meta=self._meta)
        WCSBase.__init__(
            new_copy,
            self.sip,
            (self.cpdis1, self.cpdis2),
            self.wcs,
            (self.det2im1, self.det2im2),
        )
        new_copy.__dict__.update(self.__dict__)
        return new_copy

    def __deepcopy__(self, memo: dict) -> GCRSWCS:
        """
        Return a deep copy of this WCS.

        We need to override this to pass the NormalizedMeta to the new object's constructor.
        """
        from copy import deepcopy  # noqa: PLC0415

        new_copy = self.__class__(meta=self._meta)
        new_copy.naxis = deepcopy(self.naxis, memo)
        WCSBase.__init__(
            new_copy,
            deepcopy(self.sip, memo),
            (deepcopy(self.cpdis1, memo), deepcopy(self.cpdis2, memo)),
            deepcopy(self.wcs, memo),
            (deepcopy(self.det2im1, memo), deepcopy(self.det2im2, memo)),
        )
        for key, val in self.__dict__.items():
            new_copy.__dict__[key] = deepcopy(val, memo)
        return new_copy
