import os

import numpy as np
import pytest
import reproject
from astropy.io import fits

from punchbowl.data import io
from punchbowl.level3 import celestial_intermediary

TESTDATA_DIR = os.path.dirname(__file__)
TEST_FILE = TESTDATA_DIR + '/data/downsampled_L2_PTM.fits'


def test_to_celestial_frame_cutout():
    data_cube = io.load_ndcube_from_fits(TEST_FILE, include_provenance=False)
    reprojected_cube = celestial_intermediary.to_celestial_frame_cutout(data_cube, cdelt=1)
    assert np.any(np.isfinite(reprojected_cube.data))
    assert np.any(np.isfinite(reprojected_cube.uncertainty.array))

    assert 0 <= reprojected_cube.wcs.wcs.crval[0] < 360
    assert reprojected_cube.wcs.wcs.crval[1] == 0
    assert reprojected_cube.wcs.wcs.cdelt[0] == reprojected_cube.wcs.wcs.cdelt[1]
    assert tuple(reprojected_cube.wcs.wcs.ctype) == ('RA---CAR', 'DEC--CAR')


# A point of concern is whether this shifting of reprojected images works across the RA=0 wrap point, so we test that
# well here.
@pytest.mark.parametrize('shift1,shift2,is_overlap',
                          # This is a "benign" pair, nowhere near the RA=0 point
                         [((0, 0), (-12, 5), True),
                          # This pair is right on either side of RA=0, but neither contains that point
                          ((40, 12), (-150, 0), False),
                          # The first one is at high RA and the second one straddles RA=0
                          ((40, 12), (-230, 0), True),
                          # Now one at low RA and one straddling RA=0
                          ((-150, 0), (-230, 1), True),
                         ])
def test_shift_image_onto(tmp_path, shift1, shift2, is_overlap):
    # Generate a test file, where we customize which portion of the sky it covers
    file1 = tmp_path / 'shifted_file1.fits'
    with fits.open(TEST_FILE) as hdul:
        for hdu in hdul[1:]:
            hdu.header['CRVAL1A'] += shift1[0]
            hdu.header['CRVAL2A'] += shift1[1]
            # Cut down to one Stokes component to speed up the test
            hdu.data = hdu.data[0]
        hdul.writeto(file1)

    # Generate another test file, which covers a different portion of the sky
    file2 = tmp_path / 'shifted_file2.fits'
    with fits.open(TEST_FILE) as hdul:
        for hdu in hdul[1:]:
            hdu.header['CRVAL1A'] += shift2[0]
            hdu.header['CRVAL2A'] += shift2[1]
            # Cut down to one Stokes component to speed up the test
            hdu.data = hdu.data[0]
        hdul.writeto(file2)

    cube1 = io.load_ndcube_from_fits(file1, key='A', include_provenance=False)
    cube2 = io.load_ndcube_from_fits(file2, key='A', include_provenance=False)
    reproj_data_1 = celestial_intermediary.to_celestial_frame_cutout(cube1, cdelt=.6)
    reproj_data_2 = celestial_intermediary.to_celestial_frame_cutout(cube2, cdelt=.6)

    # Just sanity-check that these are different images and different outputs, so we know the rest of the test is valid
    assert reproj_data_1.wcs.wcs.crval[0] != reproj_data_2.wcs.wcs.crval[0]
    assert reproj_data_1.data.shape != reproj_data_2.data.shape or not np.allclose(
        reproj_data_1.data, reproj_data_2.data, equal_nan=True)

    # And sanity-check that the data came out valid
    assert np.any(np.isfinite(reproj_data_1.data))
    assert np.any(np.isfinite(reproj_data_1.uncertainty.array))
    assert np.any(np.isfinite(reproj_data_2.data))
    assert np.any(np.isfinite(reproj_data_2.uncertainty.array))

    # The celestial intermediary images should be different cutouts of the sky, but on the same pixel grid. We test
    # that by aligning them first with this function, which simply adds and removes rows of pixels, and then by
    # actually reprojecting the data into the same output frame.
    d1_shifted_to_2 = celestial_intermediary.shift_image_onto(reproj_data_1, reproj_data_2)
    assert not is_overlap or np.any(np.isfinite(d1_shifted_to_2.data))
    assert not is_overlap or np.any(np.isfinite(d1_shifted_to_2.uncertainty.array))

    data = np.stack((cube1.data, cube1.uncertainty.array))
    d1_reprojected_to_2 = reproject.reproject_adaptive((data, cube1.wcs.celestial), reproj_data_2.wcs,
                                                       reproj_data_2.data.shape, return_footprint=False,
                                                       roundtrip_coords=False, boundary_mode='strict',
                                                       conserve_flux=True, center_jacobian=True)

    # We should be getting the same thing, to a part in 10^8
    np.testing.assert_allclose(d1_shifted_to_2.data, d1_reprojected_to_2[0], equal_nan=True, rtol=1e-8)
    np.testing.assert_allclose(d1_shifted_to_2.uncertainty.array, d1_reprojected_to_2[1], equal_nan=True, rtol=1e-8)

    # Now do it the other way
    d2_shifted_to_1 = celestial_intermediary.shift_image_onto(reproj_data_2, reproj_data_1)
    assert not is_overlap or np.any(np.isfinite(d2_shifted_to_1.data))

    data = np.stack((cube2.data, cube2.uncertainty.array))
    d2_reprojected_to_1 = reproject.reproject_adaptive((data, cube2.wcs.celestial), reproj_data_1.wcs,
                                                       reproj_data_1.data.shape, return_footprint=False,
                                                       roundtrip_coords=False, boundary_mode='strict',
                                                       conserve_flux=True, center_jacobian=True)

    # We should be getting the same thing, to a part in 10^8
    np.testing.assert_allclose(d2_shifted_to_1.data, d2_reprojected_to_1[0], equal_nan=True, rtol=1e-8)
    np.testing.assert_allclose(d2_shifted_to_1.uncertainty.array, d2_reprojected_to_1[1], equal_nan=True, rtol=1e-8)


def test_shift_image_onto_3d_cube(tmp_path):
    # Generate a test file, where we customize which portion of the sky it covers
    file1 = TEST_FILE

    # Generate another test file, which covers a different portion of the sky
    file2 = tmp_path / 'shifted_file2.fits'
    with fits.open(TEST_FILE) as hdul:
        for hdu in hdul[1:]:
            hdu.header['CRVAL1A'] += 10
            hdu.header['CRVAL2A'] += 5
        hdul.writeto(file2)

    cube1 = io.load_ndcube_from_fits(file1, key='A', include_provenance=False)
    cube2 = io.load_ndcube_from_fits(file2, key='A', include_provenance=False)
    reproj_data_1 = celestial_intermediary.to_celestial_frame_cutout(cube1, cdelt=.6)
    reproj_data_2 = celestial_intermediary.to_celestial_frame_cutout(cube2, cdelt=.6)

    # Just sanity-check that these are different images and different outputs, so we know the rest of the test is valid
    assert reproj_data_1.wcs.wcs.crval[0] != reproj_data_2.wcs.wcs.crval[0]
    assert reproj_data_1.data.shape != reproj_data_2.data.shape or not np.allclose(
        reproj_data_1, reproj_data_2, equal_nan=True)

    # And sanity-check that the data came out valid
    assert np.any(np.isfinite(reproj_data_1.data))
    assert np.any(np.isfinite(reproj_data_1.uncertainty.array))
    assert np.any(np.isfinite(reproj_data_2.data))
    assert np.any(np.isfinite(reproj_data_2.uncertainty.array))

    # The celestial intermediary images should be different cutouts of the sky, but on the same pixel grid. We test
    # that by aligning them first with this function, which simply adds and removes rows of pixels, and then by
    # actually reprojecting the data into the same output frame.
    d1_shifted_to_2 = celestial_intermediary.shift_image_onto(reproj_data_1, reproj_data_2)
    assert np.any(np.isfinite(d1_shifted_to_2.data))
    assert np.any(np.isfinite(d1_shifted_to_2.uncertainty.array))

    data = np.stack((cube1.data, cube1.uncertainty.array))
    d1_reprojected_to_2 = reproject.reproject_adaptive((data, cube1.wcs.celestial), reproj_data_2.wcs,
                                                       reproj_data_2.data.shape[1:], return_footprint=False,
                                                       roundtrip_coords=False, boundary_mode='strict',
                                                       conserve_flux=True, center_jacobian=True)

    # We should be getting the same thing, to a part in 10^8
    np.testing.assert_allclose(d1_shifted_to_2.data, d1_reprojected_to_2[0], equal_nan=True, rtol=1e-8)
    np.testing.assert_allclose(d1_shifted_to_2.uncertainty.array, d1_reprojected_to_2[1], equal_nan=True, rtol=1e-8)

    # Now do it the other way
    d2_shifted_to_1 = celestial_intermediary.shift_image_onto(reproj_data_2, reproj_data_1)
    assert np.any(np.isfinite(d2_shifted_to_1.data))

    data = np.stack((cube2.data, cube2.uncertainty.array))
    d2_reprojected_to_1 = reproject.reproject_adaptive((data, cube2.wcs.celestial), reproj_data_1.wcs,
                                                       reproj_data_1.data.shape[1:], return_footprint=False,
                                                       roundtrip_coords=False, boundary_mode='strict',
                                                       conserve_flux=True, center_jacobian=True)

    # We should be getting the same thing, to a part in 10^8
    np.testing.assert_allclose(d2_shifted_to_1.data, d2_reprojected_to_1[0], equal_nan=True, rtol=1e-8)
    np.testing.assert_allclose(d2_shifted_to_1.uncertainty.array, d2_reprojected_to_1[1], equal_nan=True, rtol=1e-8)


def test_shift_image_onto_fill_value(tmp_path):
    file1 = TEST_FILE

    # Generate another test file, which covers a different portion of the sky
    file2 = tmp_path / 'shifted_file2.fits'
    with fits.open(TEST_FILE) as hdul:
        for hdu in hdul[1:]:
            hdu.header['CRVAL1A'] += 10
            hdu.header['CRVAL2A'] += 10
        hdul.writeto(file2)

    cube1 = io.load_ndcube_from_fits(file1, key='A', include_provenance=False)
    cube2 = io.load_ndcube_from_fits(file2, key='A', include_provenance=False)
    reproj_data_1 = celestial_intermediary.to_celestial_frame_cutout(cube1, cdelt=.6)
    reproj_data_2 = celestial_intermediary.to_celestial_frame_cutout(cube2, cdelt=.6)

    d1_shifted_to_2 = celestial_intermediary.shift_image_onto(reproj_data_1, reproj_data_2, fill_value=np.nan)
    d2_shifted_to_1 = celestial_intermediary.shift_image_onto(reproj_data_2, reproj_data_1, fill_value=np.nan)

    assert np.any(np.isnan(d1_shifted_to_2.data))
    assert np.any(np.isnan(d2_shifted_to_1.data))

    # Make sure there are no NaNs in the input
    reproj_data_1.data[...] = np.nan_to_num(reproj_data_1.data)
    reproj_data_2.data[...] = np.nan_to_num(reproj_data_2.data)
    reproj_data_1.uncertainty.array[...] = np.nan_to_num(reproj_data_1.uncertainty.array)
    reproj_data_2.uncertainty.array[...] = np.nan_to_num(reproj_data_2.uncertainty.array)

    d1_shifted_to_2 = celestial_intermediary.shift_image_onto(reproj_data_1, reproj_data_2, fill_value=0)
    d2_shifted_to_1 = celestial_intermediary.shift_image_onto(reproj_data_2, reproj_data_1, fill_value=0)

    assert not np.any(np.isnan(d1_shifted_to_2.data))
    assert not np.any(np.isnan(d2_shifted_to_1.data))
    assert not np.any(np.isnan(d1_shifted_to_2.uncertainty.array))
    assert not np.any(np.isnan(d2_shifted_to_1.uncertainty.array))


def test_shift_image_onto_different_cdelts():
    reproj_data_1 = celestial_intermediary.to_celestial_frame_cutout(
        io.load_ndcube_from_fits(TEST_FILE, key='A', include_provenance=False), cdelt=1.2)
    reproj_data_2 = celestial_intermediary.to_celestial_frame_cutout(
        io.load_ndcube_from_fits(TEST_FILE, key='A', include_provenance=False), cdelt=1)

    with pytest.raises(ValueError, match=".*WCSes must have identical CDELTs.*"):
        celestial_intermediary.shift_image_onto(reproj_data_1, reproj_data_2)
