import os
import pathlib

import numpy as np
import pytest
from astropy.io import fits
from prefect.logging import disable_run_logger
from prefect.testing.utilities import prefect_test_harness

from punchbowl.data.punch_io import load_ndcube_from_fits, write_ndcube_to_fits
from punchbowl.data.punchcube import PUNCHCube
from punchbowl.data.tests.test_punch_io import sample_ndcube
from punchbowl.level1.flow import level1_nfi_core_flow

THIS_DIRECTORY = pathlib.Path(__file__).parent.resolve()

def test_nfi_core_flow_run(sample_ndcube):
    data_path = THIS_DIRECTORY / "data" / "PUNCH_L1_XR4_20251001120821_v0j.fits"
    sample_data = load_ndcube_from_fits(data_path)
    output = level1_nfi_core_flow([sample_data])

    assert isinstance(output[0],PUNCHCube)
