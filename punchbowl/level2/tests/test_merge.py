import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube

from punchbowl.data.tests.test_io import sample_ndcube
from punchbowl.level2.merge import merge_many_polarized_task


@pytest.fixture
def sample_data_list(sample_ndcube):
    """
    Generate a list of sample PUNCH data objects for testing
    """
    sample_pd1 = sample_ndcube((4096, 4096), code="PM1", level="1")
    sample_pd1.data[100:300, 300:400] = 1
    sample_pd1.data[500:700, 800:900] = 1
    sample_pd1.uncertainty.array[100:300, 300:400] = 1
    sample_pd1.uncertainty.array[500:700, 800:900] = 1

    sample_pd2 = sample_ndcube((4096, 4096), code="PM1", level="1")
    sample_pd2.data[100:300, 300:400] = 500
    sample_pd2.data[500:700, 800:900] = 500
    sample_pd2.uncertainty.array[100:300, 300:400] = np.inf
    sample_pd2.uncertainty.array[500:700, 800:900] = 0
    return [sample_pd1, sample_pd2]


@pytest.mark.prefect_test
def test_merge_many_task(sample_data_list):
    """
    Test the image_merge prefect flow using a test harness
    """
    trefoil_wcs = WCS("level2/data/trefoil_hdr.fits")
    trefoil_wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"
    trefoil_shape  = (4096, 4096)
    trefoil_wcs.array_shape = trefoil_shape

    output_punchdata = merge_many_polarized_task.fn(sample_data_list, trefoil_wcs)
    assert isinstance(output_punchdata, NDCube)
    assert output_punchdata.data.shape == (3, 4096, 4096)
    assert np.allclose(output_punchdata.data[0, 100:300, 300:400], 1)
    assert np.allclose(output_punchdata.data[0, 500:700, 800:900], 1)
    assert np.allclose(output_punchdata.uncertainty.array[0, 100:300, 300:400], 1)
    assert np.allclose(output_punchdata.uncertainty.array[0, 500:700, 800:900], 1)
