from datetime import datetime

import astropy.units as u
import numpy as np
import pytest
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube
from prefect import flow, get_run_logger
from remove_starfield import Starfield

from punchbowl.data import NormalizedMetadata
from punchbowl.data.wcs import calculate_celestial_wcs_from_helio, calculate_helio_wcs_from_celestial, get_p_angle
from punchbowl.level3.stellar import (
    from_celestial,
    generate_starfield_background,
    subtract_starfield_background_task,
    to_celestial,
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
    wcs.array_shape = shape

    return Starfield(starfield=starfield, wcs=wcs)


@pytest.fixture
def sample_ndcube() -> NDCube:
    def _sample_ndcube(shape: tuple, code: str = "PTM", level: str = "2") -> NDCube:
        data = np.ones(shape).astype(np.float32)
        sqrt_abs_data = np.sqrt(np.abs(data))
        uncertainty = StdDevUncertainty(np.interp(sqrt_abs_data, (sqrt_abs_data.min(), sqrt_abs_data.max()),
                                                  (0, 1)).astype(np.float32))
        wcs = WCS(naxis=3)
        wcs.ctype = "WAVE", "HPLT-TAN", "HPLN-TAN"
        wcs.cdelt = 0.2, 0.1, 0.1
        wcs.cunit = "Angstrom", "deg", "deg"
        wcs.crpix = 0, 0, 0
        wcs.crval = 5, 1, 1

        meta = NormalizedMetadata.load_template(code, level)
        meta["DATE-OBS"] = str(datetime(2024, 3, 21, 00, 00, 00))
        meta["FILEVRSN"] = "1"
        return NDCube(data=data, uncertainty=uncertainty, wcs=wcs, meta=meta)

    return _sample_ndcube


def test_from_celestial(sample_ndcube) -> NDCube:
    test_cube = sample_ndcube((3, 10, 10))
    data_solar = from_celestial(test_cube)
    expected_solar = np.full((3, 10, 10), 1.0, dtype=np.float32)
    assert isinstance(data_solar, NDCube)
    assert np.allclose(data_solar.data, expected_solar)

def test_to_celestial(sample_ndcube) -> NDCube:
    test_cube = sample_ndcube((3, 10, 10))
    data_cel = to_celestial(test_cube)
    expected_cel = np.full((3, 10, 10), 1.0, dtype=np.float32)
    assert isinstance(data_cel, NDCube)
    assert np.allclose(data_cel.data, expected_cel)
