import pathlib
from datetime import datetime, timezone

import numpy as np
import pytest
import scipy.ndimage
from astropy.wcs import WCS
from ndcube import NDCube

from punchbowl.data.tests.test_punch_io import sample_ndcube
from punchbowl.util import (
    find_first_existing_file,
    interpolate_data,
    load_mask_file,
    nan_percentile,
    nan_percentile_2d,
    parallel_sort_first_axis,
)

THIS_DIRECTORY = pathlib.Path(__file__).parent.resolve()


def test_find_first_existing_file():
    my_list = [None, NDCube(np.zeros(10),WCS()), NDCube(np.ones(10), WCS())]
    first_cube = find_first_existing_file(my_list)
    assert first_cube.data[0] == 0

def test_find_first_existing_file_raises_error_on_all_none():
    with pytest.raises(RuntimeError):
        first_cube = find_first_existing_file([None, None, None])

def test_interpolate_data(sample_ndcube):
    cube_before = sample_ndcube((10,10))
    cube_before.data[:] = 1
    cube_before.meta['DATE-OBS'] = str(datetime(2024, 1, 1, 0, 0, 0))

    cube_after = sample_ndcube((10,10))
    cube_after.data[:] = 2
    cube_after.meta['DATE-OBS'] = str(datetime(2024, 1, 2, 0, 0, 0))

    reference_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    data_interpolated = interpolate_data(cube_before, cube_after, reference_time)

    assert isinstance(data_interpolated, np.ndarray)
    assert np.all(data_interpolated == 1.5)


def test_nan_percentile():
    array = np.arange(1000, dtype=float).reshape((10, 10, 10)).transpose((1, 0, 2))[::-1]
    array[3] = np.nan
    np_result = np.nanpercentile(array.copy(), 3, axis=0)
    our_result = nan_percentile(array.copy(), 3)
    np.testing.assert_allclose(np_result, our_result)


def test_parallel_sort_first_axis():
    array = np.arange(1000, dtype=float).reshape((10, 10, 10)).transpose((1, 0, 2))[::-1]
    np_result = np.sort(array.copy(), axis=0)
    our_result = parallel_sort_first_axis(array.copy())
    np.testing.assert_allclose(np_result, our_result)

    array[3, 7] = np.nan
    our_nan_result = parallel_sort_first_axis(array.copy(), handle_nans=True)
    expected_nans = our_nan_result[-1, 7]
    assert np.all(np.isnan(expected_nans))
    expected_nans[:] = 0
    assert np.all(~np.isnan(our_nan_result))


def test_nan_percentile():
    array = np.arange(400, dtype=float).reshape((20, 20))
    array[4, 6] = np.nan
    array[12] = np.nan
    np_result = scipy.ndimage.generic_filter(array, np.nanmedian, 3, mode='constant', cval=np.nan)
    our_result = nan_percentile_2d(array.copy(), 50, 3, preserve_nans=False)
    np.testing.assert_allclose(np_result, our_result)
    our_nan_result = nan_percentile_2d(array.copy(), 50, 3, preserve_nans=True)
    np.testing.assert_equal(np.isnan(array), np.isnan(our_nan_result))


def test_load_mask_file():
    path = THIS_DIRECTORY / 'data' / "PUNCH_L1_MS3_20250311000000_v0c.bin"
    mask = load_mask_file(path)
    assert mask.dtype == np.bool
    assert mask.shape == (2048, 2048)
