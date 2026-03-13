import os

import numpy as np
from ndcube import NDCube

from punchbowl.data import punch_io
from punchbowl.level3.stellar import polarize_celestial_to_solar, polarize_solar_to_celestial

TESTDATA_DIR = os.path.dirname(__file__)
TEST_FILE = TESTDATA_DIR + '/data/downsampled_L2_PTM.fits'

def test_solar_celestial():
    data_cube = punch_io.load_ndcube_from_fits(TEST_FILE, include_provenance=False)
    data_cel = polarize_solar_to_celestial(data_cube)
    data_solar = polarize_celestial_to_solar(data_cel)

    assert isinstance(data_cel, NDCube)
    assert isinstance(data_solar, NDCube)
    assert np.allclose(data_solar.data, data_cube.data, atol=1e-30)
    assert data_cel.data.shape == data_cube.data.shape
    assert not np.shares_memory(data_cube.data, data_cel.data)
    assert not np.shares_memory(data_solar.data, data_cel.data)
