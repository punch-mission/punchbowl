import os
from datetime import datetime

import astropy
import astropy.units as u
import numpy as np
from astropy.coordinates import GCRS, EarthLocation, SkyCoord, get_sun
from astropy.time import Time
from astropy.wcs import WCS
from ndcube import NDCube
from sunpy.coordinates import frames

from punchbowl.data.meta import NormalizedMetadata
from punchbowl.data.wcs import calculate_helio_wcs_from_celestial, load_trefoil_wcs

_ROOT = os.path.abspath(os.path.dirname(__file__))


def test_sun_location():
    time_current = Time(datetime.utcnow())

    skycoord_sun = astropy.coordinates.get_sun(Time(datetime.utcnow()))

    skycoord_origin = SkyCoord(0*u.deg, 0*u.deg,
                              frame=frames.Helioprojective,
                              obstime=time_current,
                              observer='earth')

    with frames.Helioprojective.assume_spherical_screen(skycoord_origin.observer):
        skycoord_origin_celestial = skycoord_origin.transform_to(GCRS)

    with frames.Helioprojective.assume_spherical_screen(skycoord_origin.observer):
        assert skycoord_origin_celestial.separation(skycoord_sun) < 1 * u.arcsec
        assert skycoord_origin.separation(skycoord_sun) < 1 * u.arcsec


def test_wcs_many_point_2d_check():
    m = NormalizedMetadata.load_template("CTM", "2")
    date_obs = Time("2024-01-01T00:00:00", format='isot', scale='utc')
    m['DATE-OBS'] = str(date_obs)

    sun_radec = get_sun(date_obs)
    m['CRVAL1A'] = sun_radec.ra.to(u.deg).value
    m['CRVAL2A'] = sun_radec.dec.to(u.deg).value
    h = m.to_fits_header()
    d = NDCube(np.ones((4096, 4096), dtype=np.float32), WCS(h, key='A'), m)

    # we're at the center of the Earth so let's try that
    test_loc = EarthLocation.from_geocentric(0, 0, 0, unit=u.m)
    test_gcrs = SkyCoord(test_loc.get_gcrs(date_obs))

    wcs_celestial = d.wcs

    wcs_helio, _ = calculate_helio_wcs_from_celestial(wcs_celestial, date_obs, d.data.shape)

    npoints = 20
    input_coords = np.stack([
                             np.linspace(0, 4096, npoints).astype(int),
                             np.linspace(0, 4096, npoints).astype(int)], axis=1)

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
                                                     intermediate.data.lat.to(u.deg).value, 0))

    output_coords = np.array(output_coords)
    distances = np.linalg.norm(input_coords - output_coords, axis=1)
    assert np.mean(distances) < 0.1


def test_wcs_many_point_3d_check():
    m = NormalizedMetadata.load_template("PSM", "3")
    date_obs = Time("2024-01-01T00:00:00", format='isot', scale='utc')
    m['DATE-OBS'] = str(date_obs)
    sun_radec = get_sun(date_obs)
    m['CRVAL1A'] = sun_radec.ra.to(u.deg).value
    m['CRVAL2A'] = sun_radec.dec.to(u.deg).value
    h = m.to_fits_header()
    d = NDCube(np.ones((2, 4096, 4096), dtype=np.float32), WCS(h, key='A'), m)

    # we're at the center of the Earth so let's try that
    test_loc = EarthLocation.from_geocentric(0, 0, 0, unit=u.m)
    test_gcrs = SkyCoord(test_loc.get_gcrs(date_obs))

    wcs_celestial = d.wcs
    wcs_helio, _ = calculate_helio_wcs_from_celestial(wcs_celestial, date_obs, d.data.shape)

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