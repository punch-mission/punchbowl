import numpy as np
import pytest
from astropy.time import Time
from astropy.wcs import WCS
from prefect.testing.utilities import prefect_test_harness

from punchbowl.data.tests.test_io import sample_ndcube
from punchbowl.level2.resample import reproject_cube, reproject_many_flow


@pytest.fixture
def sample_punchdata_list(sample_ndcube):
    """
    Generate a list of sample PUNCH data objects for testing
    """
    sample_pd1 = sample_ndcube((50, 50))
    sample_pd2 = sample_ndcube((50, 50))
    return [sample_pd1, sample_pd2]


def test_reproject_many_flow(sample_punchdata_list):
    trefoil_wcs = WCS("level2/data/trefoil_hdr.fits")
    trefoil_wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"  # TODO: figure out why this is necessary, seems like a bug
    trefoil_shape = (128, 128)
    trefoil_wcs.array_shape = trefoil_shape
    with prefect_test_harness():
        output = reproject_many_flow(sample_punchdata_list, trefoil_wcs, trefoil_shape)
    for result in output:
        assert result.data.shape == trefoil_shape
        assert result.uncertainty.array.shape == trefoil_shape
