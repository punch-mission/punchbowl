import glob
import pathlib
from datetime import datetime

import numpy as np
import pytest
from ndcube import NDCube
from prefect.logging import disable_run_logger

from punchbowl.data.tests.test_punch_io import sample_ndcube
from punchbowl.exceptions import (
    IncorrectPolarizationStateWarning,
    IncorrectTelescopeWarning,
    InvalidDataError,
    LargeTimeDeltaWarning,
)
from punchbowl.level1.vignette import (
    correct_vignetting_task,
    generate_vignetting_calibration_nfi,
    generate_vignetting_calibration_wfi,
)

THIS_DIRECTORY = pathlib.Path(__file__).parent.resolve()


def test_check_calibration_time_delta_warning(sample_ndcube) -> None:
    """
    If the time between the data of interest and the calibration file is too great, then a warning is raised.
    """

    sample_data = sample_ndcube(shape=(10, 10), level="1")
    sample_data.meta['DATE-OBS'].value = str(datetime(2022, 2, 22, 16, 0, 1))
    vignetting_filename = THIS_DIRECTORY / "data" / "PUNCH_L1_GR1_20240222163425_v1.fits"

    with disable_run_logger():
        with pytest.warns(LargeTimeDeltaWarning):
            corrected_punchdata = correct_vignetting_task.fn(sample_data, vignetting_filename)
            assert isinstance(corrected_punchdata, NDCube)


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


def test_invalid_vignetting_file(sample_ndcube) -> None:
    """
    An invalid vignetting file should be provided. Check that an error is raised.
    """

    sample_data = sample_ndcube(shape=(10, 10), code="CR1", level="0")
    vignetting_filename = THIS_DIRECTORY / "data" / "bogus_filename.fits"

    with disable_run_logger():
        with pytest.raises(InvalidDataError):
            corrected_punchdata = correct_vignetting_task.fn(sample_data, vignetting_filename)


def test_invalid_polarization_state(sample_ndcube) -> None:
    """
    Check that a mismatch between polarization states in the vignetting function and data raises an error.
    """

    sample_data = sample_ndcube(shape=(10, 10), level="1")
    vignetting_filename = THIS_DIRECTORY / "data" / "PUNCH_L1_GR1_20240222163425_v1.fits"

    with disable_run_logger():
        with pytest.warns(IncorrectPolarizationStateWarning):
            corrected_punchdata = correct_vignetting_task.fn(sample_data, vignetting_filename)
            assert isinstance(corrected_punchdata, NDCube)


def test_invalid_telescope(sample_ndcube) -> None:
    """
    Check that a mismatch between telescopes in the vignetting function and data raises an error.
    """

    sample_data = sample_ndcube(shape=(10, 10), level="1")
    sample_data.meta['TELESCOP'].value = 'PUNCH-2'
    vignetting_filename = THIS_DIRECTORY / "data" / "PUNCH_L1_GR1_20240222163425_v1.fits"

    with disable_run_logger():
        with pytest.warns(IncorrectTelescopeWarning):
            corrected_punchdata = correct_vignetting_task.fn(sample_data, vignetting_filename)
            assert isinstance(corrected_punchdata, NDCube)


def test_invalid_data_file(sample_ndcube) -> None:
    """
    An invalid vignetting file should be provided. Check that an error is raised.
    """

    sample_data = sample_ndcube(shape=(10, 10), code="CR1", level="0")
    vignetting_filename = THIS_DIRECTORY / "non_existent_file.fits"

    with disable_run_logger():
        with pytest.raises(InvalidDataError):
            corrected_punchdata = correct_vignetting_task.fn(sample_data, vignetting_filename)


def test_vignetting_correction(sample_ndcube) -> None:
    """
    A valid vignetting file should be provided. Check that a corrected PUNCHData object is generated.
    """

    sample_data = sample_ndcube(shape=(10, 10), code="CR1", level="1")
    vignetting_filename = THIS_DIRECTORY / "data" / "PUNCH_L1_GR1_20240222163425_v1.fits"

    with disable_run_logger():
        corrected_punchdata = correct_vignetting_task.fn(sample_data, vignetting_filename)

    assert isinstance(corrected_punchdata, NDCube)

# TODO - test these more thoroughly

def test_generate_vignetting_calibration_wfi() -> None:
    """Test that vignetting calibration data is generated"""
    vignetting_data = generate_vignetting_calibration_wfi("data.dat", "mask.bin", spacecraft="2")
    assert isinstance(vignetting_data, np.ndarray)
    assert vignetting_data.shape == (2048, 2048)
    assert np.allclose(vignetting_data, 1)


def test_generate_vignetting_calibration_nfi() -> None:
    """Test that vignetting calibration data is generated"""
    vignetting_data = generate_vignetting_calibration_nfi(input_files = None,
                                                          path_mask = "mask.bin",
                                                          dark_path = "dark.fits",
                                                          polarizer = "R",
                                                          dateobs="2025-07-31T00:00:00.000",
                                                          version="0",
                                                          output_path=None)

    assert vignetting_data.shape == (2048, 2048)
    assert np.allclose(vignetting_data, 1)
