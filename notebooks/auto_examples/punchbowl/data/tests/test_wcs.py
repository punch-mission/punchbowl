import os
from datetime import datetime

import astropy
import astropy.units as u
import numpy as np
from astropy.coordinates import GCRS, EarthLocation, SkyCoord, get_sun
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from ndcube import NDCube
from sunpy.coordinates import frames

from punchbowl.data.meta import NormalizedMetadata
from punchbowl.data.wcs import (
    calculate_celestial_wcs_from_helio,
    calculate_helio_wcs_from_celestial,
    calculate_pc_matrix,
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

def test_wcs_many_point_2d_check():
    m = NormalizedMetadata.load_template("CTM", "2")
    date_obs = Time("2024-01-01T00:00:00", format='isot', scale='utc')
    m['DATE-OBS'] = str(date_obs)

    sun_radec = get_sun(date_obs)
    wcs_celestial = WCS({"CRVAL1": sun_radec.ra.to(u.deg).value,
                         "CRVAL2": sun_radec.dec.to(u.deg).value,
                         "CRPIX1": 2047.5,
                         "CRPIX2": 2047.5,
                         "CDELT1": -0.0225,
                         "CDELT2": 0.0225,
                         "CUNIT1": "deg",
                         "CUNIT2": "deg",
                         "CTYPE1": "RA---ARC",
                         "CTYPE2": "DEC--ARC"})

    # we're at the center of the Earth so let's try that
    test_loc = EarthLocation.from_geocentric(0, 0, 0, unit=u.m)
    test_gcrs = SkyCoord(test_loc.get_gcrs(date_obs))

    wcs_helio, _ = calculate_helio_wcs_from_celestial(wcs_celestial, date_obs, (4096, 4096))

    npoints = 20
    input_coords = np.stack([
                             np.linspace(0, 4096, npoints).astype(int),
                             np.linspace(0, 4096, npoints).astype(int)], axis=1)

    points_celestial = wcs_celestial.all_pix2world(input_coords, 0)
    points_helio = wcs_helio.all_pix2world(input_coords, 0)

    output_coords = []
    intermediates = []
    for c_pix, c_celestial, c_helio in zip(input_coords, points_celestial, points_helio):
        skycoord_celestial = SkyCoord(c_celestial[0] * u.deg, c_celestial[1] * u.deg,
                                      frame=GCRS,
                                      obstime=date_obs,
                                      observer=test_gcrs,
                                      obsgeoloc=test_gcrs.cartesian,
                                      obsgeovel=test_gcrs.velocity.to_cartesian(),
                                      distance=test_gcrs.hcrs.distance
                                      )

        intermediate = skycoord_celestial.transform_to(frames.Helioprojective)
        intermediates.append(intermediate)
        output_coords.append(wcs_helio.all_world2pix(intermediate.data.lon.to(u.deg).value,
                                                     intermediate.data.lat.to(u.deg).value, 0))

    output_coords = np.array(output_coords)
    print(intermediates[0].Tx.deg,  intermediates[0].Ty.deg, points_helio[0])
    distances = np.linalg.norm(input_coords - output_coords, axis=1)
    assert np.mean(distances) < 0.1


def test_wcs_many_point_3d_check():
    m = NormalizedMetadata.load_template("PSM", "3")
    date_obs = Time("2024-01-01T00:00:00", format='isot', scale='utc')
    m['DATE-OBS'] = str(date_obs)
    sun_radec = get_sun(date_obs)

    wcs_celestial = WCS({"CRVAL1": sun_radec.ra.to(u.deg).value,
                         "CRVAL2": sun_radec.dec.to(u.deg).value,
                         "CRPIX1": 2047.5,
                         "CRPIX2": 2047.5,
                         "CRPIX3": 0,
                         "CDELT1": -0.0225,
                         "CDELT2": 0.0225,
                         "CDELT3": 1.0,
                         "CUNIT1": "deg",
                         "CUNIT2": "deg",
                         "CUNIT3": "",
                         "CTYPE1": "RA---ARC",
                         "CTYPE2": "DEC--ARC",
                         "CTYPE3": "STOKES"})

    # we're at the center of the Earth so let's try that
    test_loc = EarthLocation.from_geocentric(0, 0, 0, unit=u.m)
    test_gcrs = SkyCoord(test_loc.get_gcrs(date_obs))


    wcs_helio, _ = calculate_helio_wcs_from_celestial(wcs_celestial, date_obs,(2, 4096, 4096))

    npoints = 20
    input_coords = np.stack([
                             np.linspace(0, 4096, npoints).astype(int),
                             np.linspace(0, 4096, npoints).astype(int),
                             np.ones(npoints, dtype=int),], axis=1)

    points_celestial = wcs_celestial.all_pix2world(input_coords, 0)
    points_helio = wcs_helio.all_pix2world(input_coords, 0)

    output_coords = []
    for c_pix, c_celestial, c_helio in zip(input_coords, points_celestial, points_helio):
        skycoord_celestial = SkyCoord(c_celestial[0] * u.deg, c_celestial[1] * u.deg,
                                      frame=GCRS,
                                      obstime=date_obs,
                                      observer=test_gcrs,
                                      obsgeoloc=test_gcrs.cartesian,
                                      obsgeovel=test_gcrs.velocity.to_cartesian(),
                                      distance=test_gcrs.hcrs.distance
                                      )

        intermediate = skycoord_celestial.transform_to(frames.Helioprojective)
        output_coords.append(wcs_helio.all_world2pix(intermediate.data.lon.to(u.deg).value,
                                                     intermediate.data.lat.to(u.deg).value, 2, 0))

    output_coords = np.array(output_coords)
    distances = np.linalg.norm(input_coords - output_coords, axis=1)
    assert np.nanmean(distances) < 0.1


def test_load_trefoil_wcs():
    trefoil_wcs, trefoil_shape = load_trefoil_wcs()
    assert trefoil_shape == (4096, 4096)
    assert isinstance(trefoil_wcs, WCS)


def test_helio_celestial_wcs():
    # header = fits.Header.fromtextfile(os.path.join(_ROOT, "example_header.txt"))
    #
    # date_obs = Time(header['DATE-OBS'])
    date_obs = Time("2021-06-20T00:00:00.000", format='isot', scale='utc')
    #
    wcs_helio = WCS({"CRVAL1": 0.0,
                     "CRVAL2": 0.0,
                     "CRPIX1": 2047.5,
                     "CRPIX2": 2047.5,
                     "CRPIX3": 0.0,
                     "CDELT1": 0.0225,
                     "CDELT2": 0.0225,
                     "CDELT3": 1.0,
                     "CUNIT1": "deg",
                     "CUNIT2": "deg",
                     "CTYPE1": "HPLN-ARC",
                     "CTYPE2": "HPLT-ARC",
                     "CTYPE3": "STOKES"})
    # wcs_helio = WCS(header)
    wcs_celestial = calculate_celestial_wcs_from_helio(wcs_helio, date_obs, (3, 4096, 4096))

    # wcs_celestial2 = WCS(header, key='A')

    test_loc = EarthLocation.from_geocentric(0, 0, 0, unit=u.m)
    test_gcrs = SkyCoord(test_loc.get_gcrs(date_obs))

    npoints = 20
    input_coords = np.stack([
                             np.linspace(0, 4096, npoints).astype(int),
                             np.linspace(0, 4096, npoints).astype(int),
                             np.ones(npoints, dtype=int),],
        axis=1)

    points_celestial = wcs_celestial.all_pix2world(input_coords, 0)
    points_helio = wcs_helio.all_pix2world(input_coords, 0)

    output_coords = []
    for c_pix, c_celestial, c_helio in zip(input_coords, points_celestial, points_helio):
        skycoord_celestial = SkyCoord(c_celestial[0] * u.deg, c_celestial[1] * u.deg,
                                      frame=GCRS,
                                      obstime=date_obs,
                                      observer="earth",
                                      # observer=test_gcrs,
                                      # obsgeoloc=test_gcrs.cartesian,
                                      # obsgeovel=test_gcrs.velocity.to_cartesian(),
                                      # distance=test_gcrs.hcrs.distance
                                      )

        intermediate = skycoord_celestial.transform_to(frames.Helioprojective(observer='earth', obstime=date_obs))
        # final = SkyCoord(intermediate.Tx, intermediate.Ty, frame="helioprojective", observer=intermediate.observer, obstime=intermediate.obstime)
        output_coords.append(wcs_helio.all_world2pix(intermediate.Tx.to(u.deg).value,
                                                     intermediate.Ty.to(u.deg).value,
                                                     2,
                                                     0))

    output_coords = np.array(output_coords)
    distances = np.linalg.norm(input_coords - output_coords, axis=1)

    assert np.nanmean(distances) < 0.1

from punchbowl.data.wcs import extract_crota_from_wcs, get_p_angle


def test_back_and_forth_wcs_from_celestial():
    date_obs = Time("2024-01-01T00:00:00", format='isot', scale='utc')
    sun_radec = get_sun(date_obs)
    pc = calculate_pc_matrix(0, [0.0225, 0.0225])
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
                         "PC1_1": pc[0, 0],
                         "PC1_2": pc[0, 1],
                         "PC2_1": pc[1, 0],
                         "PC2_2": pc[1, 1]
                         })

    print(extract_crota_from_wcs(wcs_celestial))
    wcs_helio, p_angle = calculate_helio_wcs_from_celestial(wcs_celestial, date_obs, (10, 10))
    print(extract_crota_from_wcs(wcs_helio))
    wcs_celestial_recovered = calculate_celestial_wcs_from_helio(wcs_helio, date_obs, (10, 10))
    print(extract_crota_from_wcs(wcs_celestial_recovered))

    assert np.allclose(wcs_celestial.wcs.crval, wcs_celestial_recovered.wcs.crval)
    assert np.allclose(wcs_celestial.wcs.crpix, wcs_celestial_recovered.wcs.crpix)
    assert np.allclose(wcs_celestial.wcs.pc, wcs_celestial_recovered.wcs.pc)
    assert np.allclose(wcs_celestial.wcs.cdelt, wcs_celestial_recovered.wcs.cdelt)


def test_back_and_forth_wcs_from_helio():
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

    print(extract_crota_from_wcs(wcs_helio))
    wcs_celestial = calculate_celestial_wcs_from_helio(wcs_helio.copy(), date_obs, (10, 10))
    print(extract_crota_from_wcs(wcs_celestial))
    wcs_helio_recovered, p_angle = calculate_helio_wcs_from_celestial(wcs_celestial.copy(), date_obs, (10, 10))
    print(extract_crota_from_wcs(wcs_helio_recovered))

    assert np.allclose(wcs_helio.wcs.crval, wcs_helio_recovered.wcs.crval)
    assert np.allclose(wcs_helio.wcs.crpix, wcs_helio_recovered.wcs.crpix)
    assert np.allclose(wcs_helio.wcs.pc, wcs_helio_recovered.wcs.pc)
    assert np.allclose(wcs_helio.wcs.cdelt, wcs_helio_recovered.wcs.cdelt)
