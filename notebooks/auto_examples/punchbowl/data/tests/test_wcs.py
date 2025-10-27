import os
from datetime import datetime
from itertools import product

import astropy
import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import GCRS, SkyCoord, get_sun
from astropy.time import Time
from astropy.wcs import WCS
from sunpy.coordinates import frames, get_earth

from punchbowl.data.wcs import (
    calculate_celestial_wcs_from_helio,
    calculate_helio_wcs_from_celestial,
    calculate_pc_matrix,
    extract_crota_from_wcs,
    load_trefoil_wcs,
)

_ROOT = os.path.abspath(os.path.dirname(__file__))


def test_sun_location():
    time_current = Time(datetime.utcnow())

    skycoord_sun = astropy.coordinates.get_sun(time_current)

    skycoord_origin = SkyCoord(0*u.deg, 0*u.deg,
                              frame=frames.Helioprojective,
                              obstime=time_current,
                              observer='earth')

    # with frames.Helioprojective.assume_spherical_screen(skycoord_origin.observer):
    skycoord_origin_celestial = skycoord_origin.transform_to(GCRS)

    # with frames.Helioprojective.assume_spherical_screen(skycoord_origin.observer):
    assert skycoord_origin_celestial.separation(skycoord_sun) < 1 * u.arcsec
    assert skycoord_origin.separation(skycoord_sun) < 1 * u.arcsec


def test_sun_location_gcrs():
    time_current = Time(datetime.utcnow())

    skycoord_sun = astropy.coordinates.get_sun(time_current)

    skycoord_origin = SkyCoord(0*u.deg, 0*u.deg,
                              frame=frames.Helioprojective,
                              obstime=time_current,
                              observer='earth')

    # with frames.Helioprojective.assume_spherical_screen(skycoord_origin.observer):
    skycoord_origin_helio = skycoord_sun.transform_to(frames.Helioprojective)

    # with frames.Helioprojective.assume_spherical_screen(skycoord_origin.observer):
    assert skycoord_origin_helio.separation(skycoord_sun) < 1 * u.arcsec
    assert skycoord_origin.separation(skycoord_sun) < 1 * u.arcsec

def test_extract_crota():
    pc = calculate_pc_matrix(-10*u.deg, [-1, 2])

    wcs_celestial = WCS(naxis=2)
    wcs_celestial.wcs.ctype = ("RA---AZP", "DEC--AZP")
    wcs_celestial.wcs.cunit = ("deg", "deg")
    wcs_celestial.wcs.cdelt = (-1, 2)
    wcs_celestial.wcs.crpix = (2048.5, 2048.5)
    wcs_celestial.wcs.crval = (0, 0)
    wcs_celestial.wcs.pc = pc
    assert np.allclose(extract_crota_from_wcs(wcs_celestial), -10 * u.deg)

def test_extract_crota_helio():
    pc = calculate_pc_matrix(10*u.deg, [0.0225, 0.0225])
    wcs_helio = WCS(naxis=2)
    wcs_helio.wcs.ctype = ("HPLN-AZP", "HPLT-AZP")
    wcs_helio.wcs.cunit = ("deg", "deg")
    wcs_helio.wcs.cdelt = (0.0225, 0.0225)
    wcs_helio.wcs.crpix = (2048.5, 2048.5)
    wcs_helio.wcs.crval = (0, 0)
    wcs_helio.wcs.pc = pc
    assert np.allclose(extract_crota_from_wcs(wcs_helio), 10 * u.deg)

def test_helio_to_celestial_wcs_many_points():
    for date_obs, crval1, crval2, crot in product(
            ["2021-03-20T00:00:00", "2021-01-20T00:00:00"],
            [-20, 10, 30],
            [-10, 0, 10],
            [-30, 0, 30] * u.deg):
        earth = get_earth(date_obs)
        wcs_helio = WCS({"CRVAL1": crval1,
                         "CRVAL2": crval2,
                         "CRPIX1": 2047.5,
                         "CRPIX2": 2047.5,
                         "NAXIS1": 4096,
                         "NAXIS2": 4096,
                         "CDELT1": 0.0225,
                         "CDELT2": 0.0225,
                         "CUNIT1": "deg",
                         "CUNIT2": "deg",
                         "CTYPE1": "HPLN-AZP",
                         "CTYPE2": "HPLT-AZP",
                         "PC1_1": np.cos(crot).value,
                         "PC1_2": -np.sin(crot).value,
                         "PC2_1": np.sin(crot).value,
                         "PC2_2": np.cos(crot).value,
                         "DATE-OBS": date_obs,
                         "MJD-OBS": Time(date_obs).mjd,
                         "DSUN_OBS": earth.radius.to_value(u.m),
                         "HGLN_OBS": earth.lon.to_value(u.deg),
                         "HGLT_OBS": earth.lat.to_value(u.deg),
                         "PV2_1": 0,
                         })

        wcs_celestial = calculate_celestial_wcs_from_helio(wcs_helio)

        npoints = 20
        xs, ys = np.meshgrid(
            np.linspace(0, wcs_helio.pixel_shape[0], npoints),
            np.linspace(0, wcs_helio.pixel_shape[1], npoints),
        )

        hp_coords = wcs_helio.pixel_to_world(xs, ys)
        # These come out claiming to be ICRS, but they're really GCRS (since it doesn't seem possible to tell a WCS
        # that it's GCRS)
        eq_coords = wcs_celestial.pixel_to_world(xs, ys)
        hp_to_eq_coords = hp_coords.transform_to('gcrs')

        eq_ra = eq_coords.ra.to_value(u.deg)
        hp_to_eq_ra = hp_to_eq_coords.ra.to_value(u.deg)
        straddles_wrap = np.nonzero(np.abs(eq_ra - hp_to_eq_ra) > 350)
        for r, c in zip(*straddles_wrap):
            if eq_ra[r, c] > 350:
                eq_ra[r, c] -= 360
            else:
                hp_to_eq_ra[r, c] -= 360

        np.testing.assert_allclose(eq_ra, hp_to_eq_ra, atol=1e-12)
        np.testing.assert_allclose(eq_coords.dec.to_value(u.deg), hp_to_eq_coords.dec.to_value(u.deg), atol=1e-12)

def test_celestial_to_helio_wcs_many_points():
    for date_obs, crval1, crval2, crot in product(
            ["2021-03-20T00:00:00", "2021-01-20T00:00:00"],
            [10, 90, 220],
            [-30, 0, 30],
            [-30, 0, 30] * u.deg):
        wcs_celestial = WCS({"CRVAL1": crval1,
                         "CRVAL2": crval2,
                         "CRPIX1": 2047.5,
                         "CRPIX2": 2047.5,
                         "NAXIS1": 4096,
                         "NAXIS2": 4096,
                         "CDELT1": -0.0225,
                         "CDELT2": 0.0225,
                         "CUNIT1": "deg",
                         "CUNIT2": "deg",
                         "CTYPE1": "RA---AZP",
                         "CTYPE2": "DEC--AZP",
                         "PC1_1": np.cos(crot).value,
                         "PC1_2": np.sin(crot).value,
                         "PC2_1": -np.sin(crot).value,
                         "PC2_2": np.cos(crot).value,
                         "DATE-OBS": date_obs,
                         "MJD-OBS": Time(date_obs).mjd,
                         "PV2_1": 0,
                         })

        wcs_helio, _ = calculate_helio_wcs_from_celestial(wcs_celestial)

        npoints = 20
        xs, ys = np.meshgrid(
            np.linspace(0, wcs_celestial.pixel_shape[0], npoints),
            np.linspace(0, wcs_celestial.pixel_shape[1], npoints),
        )

        hp_coords = wcs_helio.pixel_to_world(xs, ys)
        # These come out claiming to be ICRS, but they're really GCRS (since it doesn't seem possible to tell a WCS
        # that it's GCRS)
        eq_coords = wcs_celestial.pixel_to_world(xs, ys)
        # Make them bona-fide GCRS
        eq_coords = SkyCoord(eq_coords.ra, eq_coords.dec, frame='gcrs', obstime=date_obs)
        eq_to_hp_coords = eq_coords.transform_to('helioprojective')

        hp_tx = hp_coords.Tx.to_value(u.deg)
        eq_to_hp_tx = eq_to_hp_coords.Tx.to_value(u.deg)
        straddles_wrap = np.nonzero(np.abs(hp_tx - eq_to_hp_tx) > 350)
        for r, c in zip(*straddles_wrap):
            if hp_tx[r, c] > 170:
                hp_tx[r, c] -= 360
            else:
                eq_to_hp_tx[r, c] -= 360

        np.testing.assert_allclose(hp_tx, eq_to_hp_tx, atol=1e-12)
        np.testing.assert_allclose(hp_coords.Ty.to_value(u.deg), eq_to_hp_coords.Ty.to_value(u.deg), atol=1e-12)


def test_helio_to_celestial_wcs_many_points_3d():
    # The third dimension here is Stokes
    for date_obs, crval1, crval2, crot in product(
            ["2021-03-20T00:00:00", "2021-01-20T00:00:00"],
            [-20, 10, 30],
            [-10, 0, 10],
            [-30, 0, 30] * u.deg):
        earth = get_earth(date_obs)
        wcs_helio = WCS({"CRVAL1": crval1,
                         "CRVAL2": crval2,
                         "CRVAL3": 0,
                         "CRPIX1": 2047.5,
                         "CRPIX2": 2047.5,
                         "CRPIX3": 0,
                         "NAXIS1": 4096,
                         "NAXIS2": 4096,
                         "NAXIS3": 3,
                         "CDELT1": 0.0225,
                         "CDELT2": 0.0225,
                         "CDELT3": 1.0,
                         "CUNIT1": "deg",
                         "CUNIT2": "deg",
                         "CUNIT3": "",
                         "CTYPE1": "HPLN-AZP",
                         "CTYPE2": "HPLT-AZP",
                         "CTYPE3": "STOKES",
                         "PC1_1": np.cos(crot).value,
                         "PC1_2": -np.sin(crot).value,
                         "PC2_1": np.sin(crot).value,
                         "PC2_2": np.cos(crot).value,
                         "DATE-OBS": date_obs,
                         "MJD-OBS": Time(date_obs).mjd,
                         "DSUN_OBS": earth.radius.to_value(u.m),
                         "HGLN_OBS": earth.lon.to_value(u.deg),
                         "HGLT_OBS": earth.lat.to_value(u.deg),
                         "PV2_1": 0,
                         })

        wcs_celestial = calculate_celestial_wcs_from_helio(wcs_helio)

        npoints = 20
        xs, ys = np.meshgrid(
            np.linspace(0, wcs_helio.pixel_shape[0], npoints),
            np.linspace(0, wcs_helio.pixel_shape[1], npoints))
        zs = np.ones_like(xs)

        hp_coords = wcs_helio.pixel_to_world(xs, ys, zs)
        # These come out claiming to be ICRS, but they're really GCRS (since it doesn't seem possible to tell a WCS
        # that it's GCRS)
        eq_coords = wcs_celestial.pixel_to_world(xs, ys, zs)
        hp_to_eq_coords = hp_coords[0].transform_to('gcrs')

        eq_ra = eq_coords[0].ra.to_value(u.deg)
        hp_to_eq_ra = hp_to_eq_coords.ra.to_value(u.deg)
        straddles_wrap = np.nonzero(np.abs(eq_ra - hp_to_eq_ra) > 350)
        for r, c in zip(*straddles_wrap):
            if eq_ra[r, c] > 350:
                eq_ra[r, c] -= 360
            else:
                hp_to_eq_ra[r, c] -= 360

        np.testing.assert_allclose(eq_ra, hp_to_eq_ra, atol=1e-12)
        np.testing.assert_allclose(eq_coords[0].dec.to_value(u.deg), hp_to_eq_coords.dec.to_value(u.deg), atol=1e-12)
        np.testing.assert_array_equal(hp_coords[1], eq_coords[1])


def test_celestial_to_helio_wcs_many_points_3d():
    # The third dimension here is Stokes
    for date_obs, crval1, crval2, crot in product(
            ["2021-03-20T00:00:00", "2021-01-20T00:00:00"],
            [10, 90, 220],
            [-30, 0, 30],
            [-30, 0, 30] * u.deg):
        wcs_celestial = WCS({"CRVAL1": crval1,
                         "CRVAL2": crval2,
                         "CRVAL3": 0,
                         "CRPIX1": 2047.5,
                         "CRPIX2": 2047.5,
                         "CRPIX3": 0,
                         "NAXIS1": 4096,
                         "NAXIS2": 4096,
                         "NAXIS3": 3,
                         "CDELT1": -0.0225,
                         "CDELT2": 0.0225,
                         "CDELT3": 1.0,
                         "CUNIT1": "deg",
                         "CUNIT2": "deg",
                         "CUNIT3": "",
                         "CTYPE1": "RA---AZP",
                         "CTYPE2": "DEC--AZP",
                         "CTYPE3": "STOKES",
                         "PC1_1": np.cos(crot).value,
                         "PC1_2": np.sin(crot).value,
                         "PC2_1": -np.sin(crot).value,
                         "PC2_2": np.cos(crot).value,
                         "DATE-OBS": date_obs,
                         "MJD-OBS": Time(date_obs).mjd,
                         "PV2_1": 0,
                         })

        wcs_helio, _ = calculate_helio_wcs_from_celestial(wcs_celestial)

        npoints = 20
        xs, ys = np.meshgrid(
            np.linspace(0, wcs_celestial.pixel_shape[0], npoints),
            np.linspace(0, wcs_celestial.pixel_shape[1], npoints))
        zs = np.ones_like(xs)

        hp_coords = wcs_helio.pixel_to_world(xs, ys, zs)
        # These come out claiming to be ICRS, but they're really GCRS (since it doesn't seem possible to tell a WCS
        # that it's GCRS)
        eq_coords = wcs_celestial.pixel_to_world(xs, ys, zs)
        eq_coords = SkyCoord(eq_coords[0].ra, eq_coords[0].dec, frame='gcrs', obstime=date_obs), eq_coords[1]
        eq_to_hp_coords = eq_coords[0].transform_to('helioprojective')

        hp_tx = hp_coords[0].Tx.to_value(u.deg)
        eq_to_hp_tx = eq_to_hp_coords.Tx.to_value(u.deg)
        straddles_wrap = np.nonzero(np.abs(hp_tx - eq_to_hp_tx) > 350)
        for r, c in zip(*straddles_wrap):
            if hp_tx[r, c] > 170:
                hp_tx[r, c] -= 360
            else:
                eq_to_hp_tx[r, c] -= 360

        np.testing.assert_allclose(hp_tx, eq_to_hp_tx, atol=1e-12)
        np.testing.assert_allclose(hp_coords[0].Ty.to_value(u.deg), eq_to_hp_coords.Ty.to_value(u.deg), atol=1e-12)
        np.testing.assert_array_equal(hp_coords[1], eq_coords[1])


def test_load_trefoil_wcs():
    trefoil_wcs, trefoil_shape = load_trefoil_wcs()
    assert trefoil_shape == (4096, 4096)
    assert isinstance(trefoil_wcs, WCS)


@pytest.mark.parametrize("starting_rotation", [-140, -30, 0, 50, 100])
def test_back_and_forth_wcs_from_celestial(starting_rotation):
    date_obs = Time("2024-01-01T00:00:00", format='isot', scale='utc')
    sun_radec = get_sun(date_obs)
    wcs_celestial = WCS({"CRVAL1": sun_radec.ra.to(u.deg).value,
                         "CRVAL2": sun_radec.dec.to(u.deg).value,
                         "CRPIX1": 2047.5,
                         "CRPIX2": 2047.5,
                         "CDELT1": -0.0225,
                         "CDELT2": 0.0225,
                         "CUNIT1": "deg",
                         "CUNIT2": "deg",
                         "CTYPE1": "RA---AZP",
                         "CTYPE2": "DEC--AZP",
                         })
    pc = calculate_pc_matrix(starting_rotation, wcs_celestial.wcs.cdelt)
    wcs_celestial.wcs.pc = pc

    wcs_helio, p_angle = calculate_helio_wcs_from_celestial(wcs_celestial, date_obs, (10, 10))
    wcs_celestial_recovered = calculate_celestial_wcs_from_helio(wcs_helio, date_obs, (10, 10))

    assert np.allclose(wcs_celestial.wcs.crval, wcs_celestial_recovered.wcs.crval)
    assert np.allclose(wcs_celestial.wcs.crpix, wcs_celestial_recovered.wcs.crpix)
    assert np.allclose(wcs_celestial.wcs.pc, wcs_celestial_recovered.wcs.pc)
    assert np.allclose(wcs_celestial.wcs.cdelt, wcs_celestial_recovered.wcs.cdelt)
    assert list(wcs_celestial.wcs.ctype) == list(wcs_celestial_recovered.wcs.ctype)


@pytest.mark.parametrize("starting_rotation", [-140, -30, 0, 50, 100])
def test_back_and_forth_wcs_from_helio(starting_rotation):
    date_obs = Time("2024-01-01T00:00:00", format='isot', scale='utc')
    wcs_helio = WCS({"CRVAL1": 15.0,
                     "CRVAL2": 10.0,
                     "CRPIX1": 2047.5,
                     "CRPIX2": 2047.5,
                     "CDELT1": 0.0225,
                     "CDELT2": 0.0225,
                     "CUNIT1": "deg",
                     "CUNIT2": "deg",
                     "CTYPE1": "HPLN-ARC",
                     "CTYPE2": "HPLT-ARC"})
    pc = calculate_pc_matrix(starting_rotation, wcs_helio.wcs.cdelt)
    wcs_helio.wcs.pc = pc

    wcs_celestial = calculate_celestial_wcs_from_helio(wcs_helio.copy(), date_obs, (10, 10))
    wcs_helio_recovered, p_angle = calculate_helio_wcs_from_celestial(wcs_celestial.copy(), date_obs, (10, 10))

    assert np.allclose(wcs_helio.wcs.crval, wcs_helio_recovered.wcs.crval)
    assert np.allclose(wcs_helio.wcs.crpix, wcs_helio_recovered.wcs.crpix)
    assert np.allclose(wcs_helio.wcs.pc, wcs_helio_recovered.wcs.pc)
    assert np.allclose(wcs_helio.wcs.cdelt, wcs_helio_recovered.wcs.cdelt)
    assert list(wcs_helio_recovered.wcs.ctype) == list(wcs_helio.wcs.ctype)
