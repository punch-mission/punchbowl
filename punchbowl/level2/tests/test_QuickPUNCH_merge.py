# Core Python imports
import pytest
from pytest import fixture
import pathlib

# Third party imports
import numpy as np

from astropy.wcs import WCS

# punchbowl imports
from punchbowl.level2.QuickPUNCH_merge import reproject_array, mosaic, QuickPUNCH_merge_task

# Some test inputs
@fixture
def sample_wcs(naxis=2, crpix=(0,0), crval=(0,0), cdelt=(1, 1), ctype=("HPLN-ARC", "HPLT-ARC")) -> WCS:
    """
    Generate a WCS for testing
    """
    generated_wcs = WCS(naxis=naxis)

    generated_wcs.wcs.crpix = crpix
    generated_wcs.wcs.crval = crval
    generated_wcs.wcs.cdelt = cdelt
    generated_wcs.wcs.ctype = ctype

    return generated_wcs

@fixture
def sample_data(shape: tuple = (20,20)) -> np.ndarray:
    """
    Generate some random data for testing
    """

    return np.random.random(shape)

# core unit tests

#@pytest.mark.parametrize("input_array, input_wcs, output_wcs, output_shape",
#                         [(np.zeros([20,20]), sample_wcs, sample_wcs, (20,20))])
#def test_reproject_array(input_array, input_wcs, output_wcs, output_shape):
def test_reproject_array(sample_data, sample_wcs, output_shape=(20,20)):
    expected = sample_data
    actual = reproject_array(sample_data, sample_wcs, sample_wcs, output_shape)

    #assert np.allclose(actual, expected)
    assert actual.shape == output_shape

# task tests, 
# We also want to make sure the prefect task performs normally,
# this is much like the regression and unit tests above, but explicitly makes sure
# the Prefect part is working with history and such
@pytest.mark.prefect_test
def prefect_test(expected_regression_test_input, expected_regression_test_output):
	assert True  # you should check the history, and the regression part