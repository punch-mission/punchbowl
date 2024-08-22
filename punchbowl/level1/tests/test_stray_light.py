import pathlib

import numpy as np
import pytest

from ndcube import NDCube
from prefect.logging import disable_run_logger
from datetime import datetime

from punchbowl.data.tests.test_io import sample_ndcube
from punchbowl.exceptions import InvalidDataError
from punchbowl.level1.stray_light import remove_stray_light_task
from punchbowl.exceptions import LargeTimeDeltaWarning
from punchbowl.exceptions import IncorrectPolarizationState
from punchbowl.exceptions import IncorrectTelescope


THIS_DIRECTORY = pathlib.Path(__file__).parent.resolve()

@pytest.mark.prefect_test()
def test_check_calibration_time_delta_warning(sample_ndcube) -> None:
    """
    If the time between the data of interest and the calibration file is too great, then a warning is raised.
    """

    sample_data = sample_ndcube(shape=(10, 10))
    sample_data.meta['DATE-OBS'].value = str(datetime(2022, 2, 22, 16, 0, 1))
    stray_light_filename = THIS_DIRECTORY / "data" / "PUNCH_L1_SM1_20240222163425.fits"

    with disable_run_logger():
        with pytest.warns(LargeTimeDeltaWarning):
            corrected_punchdata = remove_stray_light_task.fn(sample_data, stray_light_filename)
            assert isinstance(corrected_punchdata, NDCube)


@pytest.mark.prefect_test()
def test_no_straylight_file(sample_ndcube) -> None:
    """
    An invalid vignetting file should be provided. Check that an error is raised.
    """

    sample_data = sample_ndcube(shape=(10, 10), code="CR1", level="0")
    straylight_filename = None

    with disable_run_logger():
        corrected_punchdata = remove_stray_light_task.fn(sample_data, straylight_filename)
        assert isinstance(corrected_punchdata, NDCube)
        assert corrected_punchdata.meta.history[0].comment == 'Stray light correction skipped'

@pytest.mark.prefect_test()
def test_invalid_stray_light_file(sample_ndcube) -> None:
    """
    An invalid stray_light file should be provided. Check that an error is raised.
    """

    sample_data = sample_ndcube(shape=(10, 10), code="CR1", level="0")
    stray_light_filename = THIS_DIRECTORY / "data" / "bogus_filename.fits"

    with disable_run_logger():
        with pytest.raises(InvalidDataError):
            corrected_punchdata = remove_stray_light_task.fn(sample_data, stray_light_filename)



@pytest.mark.prefect_test()
def test_invalid_stray_light_state(sample_ndcube) -> None:
    """
    Check that a mismatch between polarization states in the stray_light function and data raises an error.
    """

    sample_data = sample_ndcube(shape=(10, 10))
    stray_light_filename = THIS_DIRECTORY / "data" / "PUNCH_L1_SM1_20240222163425.fits"

    with disable_run_logger():
        with pytest.warns(IncorrectPolarizationState):
            corrected_punchdata = remove_stray_light_task.fn(sample_data, stray_light_filename)
            assert isinstance(corrected_punchdata, NDCube)


@pytest.mark.prefect_test()
def test_invalid_telescope(sample_ndcube) -> None:
    """
    Check that a mismatch between telescopes in the stray light function and 
    data raises an error.
    """

    sample_data = sample_ndcube(shape=(10, 10))
    sample_data.meta['TELESCOP'].value = 'PUNCH-2'
    stray_light_filename = THIS_DIRECTORY / "data" / "PUNCH_L1_SM1_20240222163425.fits"

    with disable_run_logger():
        with pytest.warns(IncorrectTelescope):
            corrected_punchdata = remove_stray_light_task.fn(sample_data, stray_light_filename)
            assert isinstance(corrected_punchdata, NDCube)

@pytest.mark.prefect_test()
def test_invalid_data_file(sample_ndcube) -> None:
    """
    An invalid vignetting file should be provided. Check that an error is 
    raised.
    """

    sample_data = sample_ndcube(shape=(10, 10), code="CR1", level="0")
    stray_light_filename = THIS_DIRECTORY / "non_existent_file.fits"

    with disable_run_logger():
        with pytest.raises(InvalidDataError):
            corrected_punchdata = remove_stray_light_task.fn(sample_data, stray_light_filename)


@pytest.mark.prefect_test()
def test_vignetting_correction(sample_ndcube) -> None:
    """
    A valid vignetting file should be provided. Check that a corrected PUNCHData object is generated.
    """

    sample_data = sample_ndcube(shape=(10, 10), code="CR1", level="0")
    stray_light_filename = THIS_DIRECTORY / "data" / "PUNCH_L1_SM1_20240222163425.fits"

    with disable_run_logger():
        corrected_punchdata = remove_stray_light_task.fn(sample_data, stray_light_filename)

    assert isinstance(corrected_punchdata, NDCube)
