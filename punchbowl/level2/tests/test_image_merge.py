# Core Python imports
# Third party imports
import pytest
from prefect.testing.utilities import prefect_test_harness

# punchbowl imports
from punchbowl.data import PUNCHData
from punchbowl.level2.image_merge import reproject_array, image_merge_flow
from punchbowl.level2.tests.test_fixtures_mosaic import sample_wcs, sample_data,\
    sample_ndcube, sample_punchdata, sample_punchdata_list


# core unit tests
@pytest.mark.parametrize("crpix, crval, cdelt",
                         [((0, 0), (0, 0), (1, 1)),
                          ((0, 0), (1, 1), (1, 1)),
                          ((1, 1), (2, 2), (1, 1))])
def test_reproject_array(sample_data, sample_wcs, crpix, crval, cdelt, output_shape=(20, 20)):
    """
    Test reproject_array usage
    """

    test_wcs = sample_wcs(crpix=crpix, crval=crval, cdelt=cdelt)
    expected = sample_data
    actual = reproject_array.fn(sample_data, test_wcs, test_wcs, output_shape)

    assert actual.shape == expected.shape


@pytest.mark.prefect_test
def test_image_merge_flow(sample_punchdata_list):
    """
    Test the image_merge prefect flow using a test harness
    """

    pd_list = sample_punchdata_list
    with prefect_test_harness():
        output_punchdata = image_merge_flow(pd_list)
        assert isinstance(output_punchdata, PUNCHData)
        assert output_punchdata.data.shape == (4096, 4096)
