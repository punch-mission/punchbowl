import astropy.units as u
import numpy as np
from astropy.wcs import WCS

from punchbowl.data import units


def test_calculate_image_pixel_area_rotated_input():
    # When we rotate the imager, the on-sky areas of the pixels shouldn't change
    rotations = [0, 30, 45, 60, 90, 120] * u.deg
    shape = (512, 512)
    area_maps = []
    for rotation in rotations:
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"
        wcs.wcs.pc = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
        wcs.wcs.crval = 0, 0
        wcs.wcs.crpix = 256, 256
        wcs.wcs.cdelt = 0.05, 0.05
        wcs.array_shape = shape

        area_maps.append(units.calculate_image_pixel_area(wcs, shape))

    for area_map in area_maps[1:]:
        np.testing.assert_allclose(area_maps[0], area_map)


def test_calculate_image_pixel_area_output_shape():
    # Test that the output data has the right shape, meaning axes probably aren't being transposed
    shape = (960, 1024)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"
    wcs.wcs.pc = np.eye(2)
    wcs.wcs.crval = 0, 0
    wcs.wcs.crpix = 512, 480
    wcs.wcs.cdelt = 0.05, 0.05
    wcs.array_shape = shape

    area_map = units.calculate_image_pixel_area(wcs, shape)
    assert area_map.shape == shape
