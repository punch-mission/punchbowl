# Core Python imports
# Third party imports
import pytest
from prefect.testing.utilities import prefect_test_harness

# punchbowl imports
from punchbowl.data import PUNCHData
from punchbowl.level2.QuickPUNCH_merge import quickpunch_merge_flow
from punchbowl.level2.tests.test_fixtures_mosaic import sample_wcs, sample_data,\
    sample_ndcube, sample_punchdata, sample_punchdata_list


@pytest.mark.prefect_test
def test_quickpunch_merge_flow(sample_punchdata_list):
    """
    Test the quickpunch_merge prefect flow using a test harness
    """

    pd_list = sample_punchdata_list
    with prefect_test_harness():
        output_punchdata = quickpunch_merge_flow(pd_list)
        assert isinstance(output_punchdata, PUNCHData)
        assert output_punchdata.data.shape == (4096, 4096)
