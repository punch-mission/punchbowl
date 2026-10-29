import os

import numpy as np

from punchbowl.data import punch_io
from punchbowl.data.punchcube import PUNCHCube
from punchbowl.data.tests.test_punch_io import sample_ndcube
from punchbowl.level3.stellar import polarize_celestial_to_solar, polarize_solar_to_celestial

TESTDATA_DIR = os.path.dirname(__file__)
TEST_FILE = TESTDATA_DIR + '/data/downsampled_L2_PTM.fits'

def test_solar_celestial():
    data_cube = punch_io.load_ndcube_from_fits(TEST_FILE, include_provenance=False)
    data_cel = polarize_solar_to_celestial(data_cube)
    data_solar = polarize_celestial_to_solar(data_cel)

    assert isinstance(data_cel, PUNCHCube)
    assert isinstance(data_solar, PUNCHCube)
    assert np.allclose(data_solar.data, data_cube.data, atol=1e-30)
    assert data_cel.data.shape == data_cube.data.shape
    assert not np.shares_memory(data_cube.data, data_cel.data)
    assert not np.shares_memory(data_solar.data, data_cel.data)


def test_solar_celestial_metadata_handling(sample_ndcube):
    cube = sample_ndcube((3,64,64), code="PIM", level="3")

    metakeys_orig = set(cube.meta.keys())

    for key in metakeys_orig:
        if cube.meta[key].datatype is float:
            cube.meta[key] = 42.
        elif cube.meta[key].datatype is int:
            cube.meta[key] = 42
        elif cube.meta[key].datatype is str:
            cube.meta[key] = "42"
        else:
            cube.meta[key] = None

    cube_celestial = polarize_solar_to_celestial(cube)
    cube_solar = polarize_celestial_to_solar(cube_celestial)

    metakeys_original = set(cube.meta.keys())
    metakeys_celestial = set(cube_celestial.meta.keys())
    metakeys_solar = set(cube_solar.meta.keys())

    assert metakeys_celestial == metakeys_solar == metakeys_original

    for metakey in metakeys_solar & metakeys_celestial & metakeys_original:
        assert cube_celestial.meta[metakey].value == cube_solar.meta[metakey].value == cube.meta[metakey].value
