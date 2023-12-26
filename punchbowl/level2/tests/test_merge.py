# Core Python imports
# Third party imports
import pytest
from astropy.wcs import WCS
from prefect.testing.utilities import prefect_test_harness

# punchbowl imports
from punchbowl.data import PUNCHData
from punchbowl.level2.merge import merge_many_task
from punchbowl.tests.test_data import sample_data_random, sample_punchdata, sample_punchdata_list, sample_wcs


@pytest.mark.prefect_test
def test_merge_many_task(sample_punchdata_list):
    """
    Test the image_merge prefect flow using a test harness
    """
    trefoil_wcs = WCS("level2/data/trefoil_hdr.fits")
    trefoil_wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"  # TODO: figure out why this is necessary, seems like a bug

    pd_list = sample_punchdata_list
    output_punchdata = merge_many_task.fn(pd_list, trefoil_wcs)
    assert isinstance(output_punchdata, PUNCHData)
    assert output_punchdata.data.shape == (50, 50)
