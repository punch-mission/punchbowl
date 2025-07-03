import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube

from punchbowl.util import find_first_existing_file


def test_find_first_existing_file():
    my_list = [None, NDCube(np.zeros(10),WCS()), NDCube(np.ones(10), WCS())]
    first_cube = find_first_existing_file(my_list)
    assert first_cube.data[0] == 0

def test_find_first_existing_file_raises_error_on_all_none():
    with pytest.raises(RuntimeError):
        first_cube = find_first_existing_file([None, None, None])
