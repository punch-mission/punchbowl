import numpy as np
import pytest
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube
from remove_starfield import Starfield

from punchbowl.data import NormalizedMetadata
from punchbowl.level3.velocity import track_flow


@pytest.fixture()
def one_data(shape: tuple = (128, 128)) -> NDCube:
    """
    Generate some random data for testing
    """
    data = np.ones(shape)

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLN-AZP", "HPLT-AZP"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.02, 0.02
    wcs.wcs.crpix = 64, 64
    wcs.wcs.crval = 0, 24.75

    meta = NormalizedMetadata(
        {"TYPECODE": "PT", "OBSCODE": "M", "LEVEL": "3", "OBSRVTRY": "0", "DATE-OBS": "2024-04-08T18:40:00"})
    return NDCube(data=data, wcs=wcs, meta=meta, uncertainty=StdDevUncertainty(np.zeros_like(data)))


# TODO - What kind of data can be tested on here? Would this work with two identical frames -> zero velocity?
def test_flow_tracking(one_data: NDCube) -> None:
    assert True
