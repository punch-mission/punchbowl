from datetime import datetime

import numpy as np
import pytest
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube

from punchbowl.data import NormalizedMetadata
from punchbowl.level3.low_noise import create_low_noise_task


@pytest.fixture
def sample_ndcube_polarized() -> NDCube:
    def _sample_ndcube(shape: tuple, code: str = "PTM", level: str = "3") -> NDCube:
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


@pytest.fixture
def sample_ndcube_clear() -> NDCube:
    def _sample_ndcube(shape: tuple, code: str = "CTM", level: str = "3") -> NDCube:
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


def test_low_noise_polarized(sample_ndcube_polarized) -> NDCube:
    cube_low_noise = create_low_noise_task([sample_ndcube_polarized((2,10,10)),
                                            sample_ndcube_polarized((2,10,10)),
                                            sample_ndcube_polarized((2,10,10)),
                                            sample_ndcube_polarized((2,10,10))])
    assert isinstance(cube_low_noise, NDCube)
    assert cube_low_noise.data.shape == (2,10,10)
    # TODO - other assertions


def test_low_noise_clear(sample_ndcube_clear) -> NDCube:
    cube_low_noise = create_low_noise_task([sample_ndcube_clear((10,10)),
                                            sample_ndcube_clear((10,10)),
                                            sample_ndcube_clear((10,10)),
                                            sample_ndcube_clear((10,10))])
    assert isinstance(cube_low_noise, NDCube)
    assert cube_low_noise.data.shape == (10,10)
    # TODO - other assertions
