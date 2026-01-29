import os

from punchbowl.auto.control.util import load_quicklook_scaling

TESTDATA_DIR = os.path.dirname(__file__)

def test_load_quicklook_scaling():
    vmin, vmax = load_quicklook_scaling(level="0", product="CR", obscode="2", path=TESTDATA_DIR +"/punchpipe_config.yaml")

    assert vmin == 100
    assert vmax == 800


def test_load_quicklook_scaling_no_input():
    vmin, vmax = load_quicklook_scaling(path=TESTDATA_DIR+"/punchpipe_config.yaml")

    assert vmin == 5e-13
    assert vmax == 5e-11


def test_load_quicklook_scaling_no_product():
    vmin, vmax = load_quicklook_scaling(level="0", path=TESTDATA_DIR+"/punchpipe_config.yaml")

    assert vmin == 100
    assert vmax == 800
