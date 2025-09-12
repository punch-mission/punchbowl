import os
from datetime import datetime

import numpy as np

from punchbowl.cli import create_calibration
from punchbowl.data import load_ndcube_from_fits

# TODO - test these more thoroughly

def test_vignetting_creation_wfi(tmpdir):
    with open("empty.txt", "w") as f:
        f.write("tappin.dat\nmask.bin\n")
    create_calibration(level = "1",
                       code = "GR",
                       spacecraft = "2",
                       timestamp = datetime(2025, 7, 2, 12, 0, 0),
                       file_version = "0",
                       input_list_path = "empty.txt",
                       input_files = None,
                       out_path = str(tmpdir))

    expected_path = os.path.join(str(tmpdir), "PUNCH_L1_GR2_20250702120000_v0.fits")
    cube = load_ndcube_from_fits(expected_path)
    assert True
    assert np.allclose(cube.data, 1)


def test_vignetting_creation_nfi(tmpdir):
    with open("empty.txt", "w") as f:
        f.write("dark.fits\nmask.bin\n")
    create_calibration(level = "1",
                       code = "GR",
                       spacecraft = "4",
                       timestamp = datetime(2025, 7, 2, 12, 0, 0),
                       file_version = "0",
                       input_list_path = "empty.txt",
                       input_files = None,
                       out_path = str(tmpdir))

    expected_path = os.path.join(str(tmpdir), "PUNCH_L1_GR4_20250702120000_v0.fits")
    cube = load_ndcube_from_fits(expected_path)
    assert np.allclose(cube.data, 1)
