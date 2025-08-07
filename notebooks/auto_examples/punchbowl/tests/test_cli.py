import os
from datetime import datetime

import numpy as np

from punchbowl.cli import create_calibration
from punchbowl.data import load_ndcube_from_fits


def test_vignetting_creation_returns_ones_for_nfi(tmpdir):
    with open("empty.txt", "w") as f:
        f.write("tappin.dat\nmask.bin\n")
    create_calibration("1",
                       "GR",
                       "4",
                       datetime(2025, 7, 2, 12, 0, 0),
                       "0",
                       "empty.txt",
                       str(tmpdir))

    expected_path = os.path.join(str(tmpdir), "PUNCH_L1_GR4_20250702120000_v0.fits")
    cube = load_ndcube_from_fits(expected_path)
    assert np.allclose(cube.data, 1)
