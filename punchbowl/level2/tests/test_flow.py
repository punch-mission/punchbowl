import os
import pathlib

import pytest
from astropy.wcs import WCS
from ndcube import NDCube
from prefect.testing.utilities import prefect_test_harness

from punchbowl.data.io import write_ndcube_to_fits
from punchbowl.data.tests.test_io import sample_ndcube
from punchbowl.level2.flow import ORDER, level2_core_flow

THIS_DIRECTORY = pathlib.Path(__file__).parent.resolve()

def test_core_flow_runs_with_filenames(sample_ndcube, tmpdir):
    data_list = [sample_ndcube(shape=(10, 10), code=code, level="1") for code in ORDER]
    paths = []
    for i, cube in enumerate(data_list):
        path = os.path.join(tmpdir, f"test_input_{i}.fits")
        write_ndcube_to_fits(cube, path)
        paths.append(path)
    voters = [[] for _ in data_list]

    with prefect_test_harness():
        output = level2_core_flow(paths, voters, trefoil_wcs=WCS(naxis=2), trefoil_shape=(20, 20))
    assert isinstance(output[0], NDCube)

@pytest.mark.parametrize("drop_indices", [[], [1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
def test_core_flow_runs_with_objects_and_calibration_files(sample_ndcube, drop_indices):
    data_list = [sample_ndcube(shape=(10, 10), code=code, level="1")
                 for i, code in enumerate(ORDER) if i not in drop_indices]

    voters = [[] for _ in data_list]
    with prefect_test_harness():
        output = level2_core_flow(data_list, voters, trefoil_wcs=WCS(naxis=2), trefoil_shape=(20, 20))
    assert isinstance(output[0], NDCube)
