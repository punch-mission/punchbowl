import numpy as np
import pytest
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube
from remove_starfield import Starfield

from punchbowl.data import NormalizedMetadata
from punchbowl.level3.stellar import (
    generate_starfield_background,
    subtract_starfield_background,
    subtract_starfield_background_task,
)


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
        {"TYPECODE": "CS", "OBSCODE": "M", "LEVEL": "3", "OBSRVTRY": "0", "DATE-OBS": "2024-04-08T18:40:00"})
    return NDCube(data=data, wcs=wcs, meta=meta, uncertainty=StdDevUncertainty(np.zeros_like(data)))


@pytest.fixture()
def zero_starfield_data(shape: tuple = (256, 256)) -> Starfield:
    """
    Generate some random data for testing
    """
    starfield = np.zeros(shape)

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLN-AZP", "HPLT-AZP"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.02, 0.02
    wcs.wcs.crpix = 128, 128
    wcs.wcs.crval = 0, 24.75

    return Starfield(starfield=starfield, wcs=wcs)


def test_basic_subtraction(one_data: NDCube, zero_starfield_data: Starfield) -> None:
    """

    """
    subtraction_starfield = subtract_starfield_background(one_data, zero_starfield_data)
    subtraction_punchdata = NDCube(data=subtraction_starfield.data, wcs=subtraction_starfield.wcs, meta=subtraction_starfield.meta)
    assert isinstance(subtraction_punchdata, NDCube)
    assert np.allclose(subtraction_punchdata.data, 1)
