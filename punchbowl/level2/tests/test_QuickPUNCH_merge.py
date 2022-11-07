# Core Python imports
import pytest  
from pytest import fixture  
import pathlib  

# Third party imports
import numpy as np  

# punchbowl imports
from punchbowl.level1.this_module import supporting_function, module_task  

# core unit tests
@pytest.mark.parametrize("diag, above, below",  
                         [(5, 4, 3),  
                          (5, 1, 1)])  
def underscore_descriptive_name(variable1, variable2):  
    expected = np.zeros(5)  # replace this with whatever you expect
    actual = supporting_function(variable1, variable2)

	# Now do your assertions! Whatever tests that your code works. 
    assert np.allclose(actual, expected)  
    assert actual.shape == (4, 4)  

# parameterized example
@pytest.mark.parametrize("variable1, variable2, expected",  
                         [(5, 1, 2),  
                          (100, 1, 2)])  
def other_descriptive_name(variable1, variable2, expected):  
    actual = supporting_function(variable1, variable2)

	# Now do your assertions! Whatever tests that your code works. 
    assert np.allclose(actual, expected)  

# Don't forget regression tests on real data! We mark those as regression using
# pytest.mark so that we don't have to always run them

# here we make a test input, this should be real data
@pytest.mark.regression  
@fixture  
def regression_test_input() -> np.ndarray:  
    """
    If it's non-trivial, explain what this data is!
    """
    path = "what data I'm loading"
    return np.load(str(path))  # load it with whatever tool you'd normally do


# here we make the expected outcome
@pytest.mark.regression  
@fixture  
def expected_regression_test_output() -> np.ndarray:
    path = TEST_DIRECTORY / "data/regression_output_destreak.npy"  
    return np.load(str(path))  


# now for the actual regression test
@pytest.mark.regression  
def test_regression(expected_regression_test_input, expected_regression_test_output):  
    test_output = supporting_function(expected_regression_test_input)  
    assert np.allclose(test_output, expected_regression_test_output)  # validate that it's close to what you normally get

# task tests, 
# We also want to make sure the prefect task performs normally,
# this is much like the regression and unit tests above, but explicitly makes sure
# the Prefect part is working with history and such
@pytest.mark.prefect_test
def prefect_test(expected_regression_test_input, expected_regression_test_output):
	assert True  # you should check the history, and the regression part