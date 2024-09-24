import pathlib
from datetime import datetime

import numpy as np
import pytest
from ndcube import NDCube
from prefect.logging import disable_run_logger

from punchbowl.data.tests.test_io import sample_ndcube
from punchbowl.exceptions import LargeTimeDeltaWarning
#from punchbowl.exceptions import InvalidDataError
from punchbowl.level1.stray_light import remove_stray_light_task

THIS_DIRECTORY = pathlib.Path(__file__).parent.resolve()

@pytest.mark.prefect_test()
def test_check_calibration_time_delta_warning(sample_ndcube) -> None:
    """
    If the time between the data of interest and the calibration file is too great, then a warning is raised.
    """

    sample_data = sample_ndcube(shape=(10, 10))
    sample_data.meta['DATE-OBS'].value = str(datetime(2022, 2, 22, 16, 0, 1))
    stray_light_filename = THIS_DIRECTORY / "data" / "PUNCH_L1_SM1_20240222163425_v1.fits"

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
