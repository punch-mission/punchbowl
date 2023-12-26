# Core Python imports
# Third party imports
import pytest
from astropy.wcs import WCS
from prefect.testing.utilities import prefect_test_harness

from punchbowl.level2.resample import reproject_array, reproject_many_flow
# punchbowl imports
from punchbowl.tests.test_data import sample_data_random, sample_punchdata, sample_punchdata_list, sample_wcs


@pytest.mark.parametrize("crpix, crval, cdelt",
                         [((0, 0), (0, 0), (1, 1)),
                          ((0, 0), (1, 1), (1, 1)),
                          ((1, 1), (2, 2), (1, 1))])
def test_reproject_array(sample_data_random, sample_wcs, crpix, crval, cdelt, output_shape=(50, 50)):
    """
    Test reproject_array usage
    """

    test_wcs = sample_wcs(crpix=crpix, crval=crval, cdelt=cdelt)
    expected = sample_data_random
    actual = reproject_array.fn(sample_data_random, test_wcs, test_wcs, output_shape)

    assert actual.shape == expected.shape


def test_reproject_many_flow(sample_punchdata_list):
    trefoil_wcs = WCS("level2/data/trefoil_hdr.fits")
    trefoil_wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"  # TODO: figure out why this is necessary, seems like a bug

    trefoil_shape = (128, 128)
    with prefect_test_harness():
        output = reproject_many_flow(sample_punchdata_list, trefoil_wcs, trefoil_shape)
    for result in output:
        assert result.data.shape == trefoil_shape
        assert result.uncertainty.array.shape == trefoil_shape
