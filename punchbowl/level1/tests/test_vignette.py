# Core Python imports
import pathlib

# Third party imports
import numpy as np
import pytest
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from prefect.logging import disable_run_logger

# punchbowl imports
from punchbowl.data import NormalizedMetadata, PUNCHData
from punchbowl.exceptions import InvalidDataError
from punchbowl.level1.vignette import correct_vignetting_task
from punchbowl.tests.test_data import sample_data_random, sample_punchdata, sample_punchdata_clear

THIS_DIRECTORY = pathlib.Path(__file__).parent.resolve()


@pytest.mark.prefect_test()
def test_invalid_vignetting_file(sample_punchdata_clear: PUNCHData) -> None:
    """
    An invalid vignetting file should be provided. Check that an error is raised.
    """

    sample_data = sample_punchdata_clear(shape=(10, 10))
    vignetting_filename = THIS_DIRECTORY / "data" / "bogus_filename.fits"

    with disable_run_logger():
        with pytest.raises(InvalidDataError):
            corrected_punchdata = correct_vignetting_task.fn(sample_data, vignetting_filename)

@pytest.mark.prefect_test()
def test_invalid_polarization_state(sample_punchdata: PUNCHData) -> None:
    """
    Check that a mismatch bewteen polarization states in the vignetting function and data raises an error.
    """

    sample_data = sample_punchdata(shape=(10, 10))
    vignetting_filename = THIS_DIRECTORY / "data" / "test_vignetting_function.fits"

    with disable_run_logger():
        with pytest.raises(InvalidDataError):
            corrected_punchdata = correct_vignetting_task.fn(sample_data, vignetting_filename)


@pytest.mark.prefect_test()
def test_vignetting_correction(sample_punchdata_clear: PUNCHData) -> None:
    """
    A valid vignetting file should be provided. Check that a corrected PUNCHData object is generated.
    """

    sample_data = sample_punchdata_clear(shape=(10, 10))
    vignetting_filename = THIS_DIRECTORY / "data" / "test_vignetting_function.fits"

    with disable_run_logger():
        corrected_punchdata = correct_vignetting_task.fn(sample_data, vignetting_filename)

    assert isinstance(corrected_punchdata, PUNCHData)
