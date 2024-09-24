import numpy as np
import pytest
from astropy.time import Time
from astropy.wcs import WCS
from prefect.testing.utilities import prefect_test_harness

from punchbowl.data.tests.test_io import sample_ndcube
from punchbowl.level2.resample import reproject_array, reproject_many_flow


@pytest.fixture
def sample_punchdata_list(sample_ndcube):
    """
    Generate a list of sample PUNCHData objects for testing
    """
    sample_pd1 = sample_ndcube((50, 50))
    sample_pd2 = sample_ndcube((50, 50))
    return [sample_pd1, sample_pd2]


@pytest.fixture
def sample_wcs() -> WCS:
    """Generate a sample WCS for testing."""
    def _sample_wcs(naxis=2, crpix=(0, 0), crval=(0, 0), cdelt=(1, 1),
                    ctype=("HPLN-ARC", "HPLT-ARC")):
        generated_wcs = WCS(naxis=naxis)

        generated_wcs.wcs.crpix = crpix
        generated_wcs.wcs.crval = crval
        generated_wcs.wcs.cdelt = cdelt
        generated_wcs.wcs.ctype = ctype

        return generated_wcs
    return _sample_wcs


@pytest.mark.parametrize("crpix, crval, cdelt",
                         [((0, 0), (0, 0), (1, 1)),
                          ((0, 0), (1, 1), (1, 1)),
                          ((1, 1), (2, 2), (1, 1))])
def test_reproject_array(sample_wcs, crpix, crval, cdelt, output_shape=(50, 50)):
    """
    Test reproject_array usage
    """
    shape = (50, 50)
    test_wcs = sample_wcs(crpix=crpix, crval=crval, cdelt=cdelt)
    expected = np.random.random(shape)
    now = Time.now()
    actual = reproject_array.fn(expected, test_wcs, now, test_wcs, output_shape)

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
