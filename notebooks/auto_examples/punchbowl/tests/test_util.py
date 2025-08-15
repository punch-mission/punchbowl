from datetime import datetime, timezone

import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube

from punchbowl.data.tests.test_punch_io import sample_ndcube
from punchbowl.util import find_first_existing_file, interpolate_data


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
