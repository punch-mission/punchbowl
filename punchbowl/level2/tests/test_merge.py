import pytest
from astropy.wcs import WCS
from ndcube import NDCube

from punchbowl.data.tests.test_io import sample_ndcube
from punchbowl.level2.merge import merge_many_polarized_task


@pytest.fixture
def sample_data_list(sample_ndcube):
    """
    Generate a list of sample PUNCHData objects for testing
    """
    sample_pd1 = sample_ndcube((4096, 4096))
    sample_pd2 = sample_ndcube((4096, 4096))
    return [sample_pd1, sample_pd2]


@pytest.mark.prefect_test
def test_merge_many_task(sample_data_list):
    """
    Test the image_merge prefect flow using a test harness
    """
    trefoil_wcs = WCS("level2/data/trefoil_hdr.fits")
    trefoil_wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"  # TODO: figure out why this is necessary, seems like a bug

    pd_list = sample_data_list
    output_punchdata = merge_many_polarized_task.fn(pd_list, trefoil_wcs)
    assert isinstance(output_punchdata, NDCube)
    assert output_punchdata.data.shape == (3, 4096, 4096)
