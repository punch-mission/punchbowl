import pathlib
from datetime import datetime

import numpy as np
import pytest
from ndcube import NDCube
from prefect.logging import disable_run_logger

from punchbowl.data.tests.test_io import sample_ndcube
from punchbowl.exceptions import (
    IncorrectPolarizationStateWarning,
    IncorrectTelescopeWarning,
    InvalidDataError,
    LargeTimeDeltaWarning,
)
from punchbowl.level1.vignette import correct_vignetting_task

THIS_DIRECTORY = pathlib.Path(__file__).parent.resolve()


@pytest.mark.prefect_test()
def test_check_calibration_time_delta_warning(sample_ndcube) -> None:
    """
    If the time between the data of interest and the calibration file is too great, then a warning is raised.
    """

    sample_data = sample_ndcube(shape=(10, 10))
    sample_data.meta['DATE-OBS'].value = str(datetime(2022, 2, 22, 16, 0, 1))
    vignetting_filename = THIS_DIRECTORY / "data" / "PUNCH_L1_GR1_20240222163425_v1.fits"

    with disable_run_logger():
        with pytest.warns(LargeTimeDeltaWarning):
            corrected_punchdata = correct_vignetting_task.fn(sample_data, vignetting_filename)
            assert isinstance(corrected_punchdata, NDCube)


@pytest.mark.prefect_test()
def test_no_vignetting_file(sample_ndcube) -> None:
    """
    An invalid vignetting file should be provided. Check that an error is raised.
    """

    sample_data = sample_ndcube(shape=(10, 10), code="CR1", level="0")
    vignetting_filename = None

    with disable_run_logger():
        corrected_punchdata = correct_vignetting_task.fn(sample_data, vignetting_filename)
        assert isinstance(corrected_punchdata, NDCube)
        assert corrected_punchdata.meta.history[0].comment == 'Vignetting skipped'


@pytest.mark.prefect_test()
def test_invalid_vignetting_file(sample_ndcube) -> None:
    """
    An invalid vignetting file should be provided. Check that an error is raised.
    """

    sample_data = sample_ndcube(shape=(10, 10), code="CR1", level="0")
    vignetting_filename = THIS_DIRECTORY / "data" / "bogus_filename.fits"

    with disable_run_logger():
        with pytest.raises(InvalidDataError):
            corrected_punchdata = correct_vignetting_task.fn(sample_data, vignetting_filename)


@pytest.mark.prefect_test()
def test_invalid_polarization_state(sample_ndcube) -> None:
    """
    Check that a mismatch between polarization states in the vignetting function and data raises an error.
    """

    sample_data = sample_ndcube(shape=(10, 10))
    vignetting_filename = THIS_DIRECTORY / "data" / "PUNCH_L1_GR1_20240222163425_v1.fits"

    with disable_run_logger():
        with pytest.warns(IncorrectPolarizationStateWarning):
            corrected_punchdata = correct_vignetting_task.fn(sample_data, vignetting_filename)
            assert isinstance(corrected_punchdata, NDCube)


@pytest.mark.prefect_test()
def test_invalid_telescope(sample_ndcube) -> None:
    """
    Check that a mismatch between telescopes in the vignetting function and data raises an error.
    """

    sample_data = sample_ndcube(shape=(10, 10))
    sample_data.meta['TELESCOP'].value = 'PUNCH-2'
    vignetting_filename = THIS_DIRECTORY / "data" / "PUNCH_L1_GR1_20240222163425_v1.fits"

    with disable_run_logger():
        with pytest.warns(IncorrectTelescopeWarning):
            corrected_punchdata = correct_vignetting_task.fn(sample_data, vignetting_filename)
            assert isinstance(corrected_punchdata, NDCube)


@pytest.mark.prefect_test()
def test_invalid_data_file(sample_ndcube) -> None:
    """
    An invalid vignetting file should be provided. Check that an error is raised.
    """

    sample_data = sample_ndcube(shape=(10, 10), code="CR1", level="0")
    vignetting_filename = THIS_DIRECTORY / "non_existent_file.fits"

    with disable_run_logger():
        with pytest.raises(InvalidDataError):
            corrected_punchdata = correct_vignetting_task.fn(sample_data, vignetting_filename)


@pytest.mark.prefect_test()
def test_vignetting_correction(sample_ndcube) -> None:
    """
    A valid vignetting file should be provided. Check that a corrected PUNCHData object is generated.
    """

    sample_data = sample_ndcube(shape=(10, 10), code="CR1", level="0")
    vignetting_filename = THIS_DIRECTORY / "data" / "PUNCH_L1_GR1_20240222163425_v1.fits"

    with disable_run_logger():
        corrected_punchdata = correct_vignetting_task.fn(sample_data, vignetting_filename)

    assert isinstance(corrected_punchdata, NDCube)
