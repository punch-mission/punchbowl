import pathlib

import numpy as np
import pytest
from ndcube import NDCube
from prefect.logging import disable_run_logger

from punchbowl.exceptions import InvalidDataError
from punchbowl.level1.vignette import correct_vignetting_task
from punchbowl.tests.test_data import sample_data_random, sample_punchdata, sample_punchdata_clear

THIS_DIRECTORY = pathlib.Path(__file__).parent.resolve()


@pytest.mark.prefect_test()
def test_no_vignetting_file(sample_punchdata_clear: NDCube) -> None:
    """
    An invalid vignetting file should be provided. Check that an error is raised.
    """

    sample_data = sample_punchdata_clear(shape=(10, 10))
    vignetting_filename = None

    with disable_run_logger():
        corrected_punchdata = correct_vignetting_task.fn(sample_data, vignetting_filename)
        assert isinstance(corrected_punchdata, NDCube)
        assert corrected_punchdata.meta.history[0].comment == 'Vignetting skipped'


@pytest.mark.prefect_test()
def test_invalid_vignetting_file(sample_punchdata_clear: NDCube) -> None:
    """
    An invalid vignetting file should be provided. Check that an error is raised.
    """

    sample_data = sample_punchdata_clear(shape=(10, 10))
    vignetting_filename = THIS_DIRECTORY / "data" / "bogus_filename.fits"

    with disable_run_logger():
        with pytest.raises(InvalidDataError):
            corrected_punchdata = correct_vignetting_task.fn(sample_data, vignetting_filename)

@pytest.mark.prefect_test()
def test_invalid_polarization_state(sample_punchdata: NDCube) -> None:
    """
    Check that a mismatch between polarization states in the vignetting function and data raises an error.
    """

    sample_data = sample_punchdata(shape=(10, 10))
    vignetting_filename = THIS_DIRECTORY / "data" / "PUNCH_L1_GR1_20240222163425.fits"

    with disable_run_logger():
        with pytest.warns(UserWarning):
            corrected_punchdata = correct_vignetting_task.fn(sample_data, vignetting_filename)
            assert isinstance(corrected_punchdata, NDCube)


@pytest.mark.prefect_test()
def test_invalid_telescope(sample_punchdata: NDCube) -> None:
    """
    Check that a mismatch between telescopes in the vignetting function and data raises an error.
    """

    sample_data = sample_punchdata(shape=(10, 10))
    sample_data.meta['TELESCOP'].value = 'PUNCH-2'
    vignetting_filename = THIS_DIRECTORY / "data" / "PUNCH_L1_GR1_20240222163425.fits"

    with disable_run_logger():
        with pytest.warns(UserWarning):
            corrected_punchdata = correct_vignetting_task.fn(sample_data, vignetting_filename)
            assert isinstance(corrected_punchdata, NDCube)


@pytest.mark.prefect_test()
def test_invalid_data_file(sample_punchdata_clear: NDCube) -> None:
    """
    An invalid vignetting file should be provided. Check that an error is raised.
    """

    sample_data = sample_punchdata_clear(shape=(20, 20))
    vignetting_filename = THIS_DIRECTORY / "data" / "PUNCH_L1_GR1_20240222163425.fits"

    with disable_run_logger():
        with pytest.raises(InvalidDataError):
            corrected_punchdata = correct_vignetting_task.fn(sample_data, vignetting_filename)


@pytest.mark.prefect_test()
def test_vignetting_correction(sample_punchdata_clear: NDCube) -> None:
    """
    A valid vignetting file should be provided. Check that a corrected PUNCHData object is generated.
    """

    sample_data = sample_punchdata_clear(shape=(10, 10))
    vignetting_filename = THIS_DIRECTORY / "data" / "PUNCH_L1_GR1_20240222163425.fits"

    with disable_run_logger():
        corrected_punchdata = correct_vignetting_task.fn(sample_data, vignetting_filename)

    assert isinstance(corrected_punchdata, NDCube)
