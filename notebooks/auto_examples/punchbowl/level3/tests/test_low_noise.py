from datetime import datetime

import numpy as np
import pytest
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS

from punchbowl.data import NormalizedMetadata
from punchbowl.data.punchcube import PUNCHCube
from punchbowl.level3 import low_noise


@pytest.fixture
def sample_ndcube_polarized() -> PUNCHCube:
    def _sample_ndcube(shape: tuple, code: str = "PTM", level: str = "3") -> PUNCHCube:
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
        meta["DATE-OBS"] = str(datetime(2024, 3, 21, 1, 00, 00))
        meta["DATE-BEG"] = str(datetime(2024, 3, 21, 00, 59, 30))
        meta["DATE-END"] = str(datetime(2024, 3, 21, 1, 00, 30))
        meta["FILEVRSN"] = "1"
        return PUNCHCube(data=data, uncertainty=uncertainty, wcs=wcs, meta=meta)

    return _sample_ndcube


@pytest.fixture
def sample_ndcube_clear() -> PUNCHCube:
    def _sample_ndcube(shape: tuple, code: str = "CTM", level: str = "3") -> PUNCHCube:
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
        meta["DATE-OBS"] = str(datetime(2024, 3, 21, 1, 00, 00))
        meta["DATE-BEG"] = str(datetime(2024, 3, 21, 00, 59, 30))
        meta["DATE-END"] = str(datetime(2024, 3, 21, 1, 00, 30))
        meta["FILEVRSN"] = "1"
        return PUNCHCube(data=data, uncertainty=uncertainty, wcs=wcs, meta=meta)

    return _sample_ndcube


def test_low_noise_polarized(sample_ndcube_polarized) -> PUNCHCube:
    cube_low_noise = low_noise.create_low_noise_task([sample_ndcube_polarized((3,10,10)),
                                            sample_ndcube_polarized((3,10,10)),
                                            sample_ndcube_polarized((3,10,10)),
                                            sample_ndcube_polarized((3,10,10))])
    assert isinstance(cube_low_noise, PUNCHCube)
    assert cube_low_noise.data.shape == (3,10,10)
    assert cube_low_noise.meta.product_code == "PAM"


def test_low_noise_clear(sample_ndcube_clear) -> PUNCHCube:
    cube_low_noise = low_noise.create_low_noise_task([sample_ndcube_clear((10,10)),
                                            sample_ndcube_clear((10,10)),
                                            sample_ndcube_clear((10,10)),
                                            sample_ndcube_clear((10,10))])
    assert isinstance(cube_low_noise, PUNCHCube)
    assert cube_low_noise.data.shape == (10,10)
    assert cube_low_noise.meta.product_code == "CAM"


@pytest.mark.parametrize("signals, errors, mean, sigma", [
    ([1, -1], [2, 2], 0, 2),
    ([1, 3], [2, 4], 1.4, np.sqrt(10)),
    ([0, 1], [1, 2], 0.2, np.sqrt(2.5)),
])
def test_pam_weighted_uncertainty(sample_ndcube_polarized, signals, errors, mean, sigma):
    """PAM keeps signed and zero signals and uses the shared RMS error convention."""
    cubes = [sample_ndcube_polarized((3, 2, 2)) for _ in signals]
    for cube, signal, error in zip(cubes, signals, errors):
        cube.data[:] = signal
        cube.uncertainty.array[:] = error
    result = low_noise.create_low_noise_task.fn(cubes)
    np.testing.assert_allclose(result.data, mean)
    np.testing.assert_allclose(result.uncertainty.array, sigma)
    assert result.meta.product_code == "PAM"


def test_pam_invalid_samples(sample_ndcube_polarized):
    """Invalid data/errors and masked samples contribute no weight to PAM."""
    good = sample_ndcube_polarized((3, 2, 4))
    data = np.broadcast_to([[np.nan, np.inf, 50, 50], [50, 50, 50, 50]], good.shape).copy()
    sigma = np.broadcast_to([[1, 1, 0, -1], [np.nan, np.inf, 1, 1]], good.shape).copy()
    mask = np.zeros(good.shape, dtype=bool)
    mask[:, 1, 2:] = True
    bad = good.replace(data=data, uncertainty=StdDevUncertainty(sigma), mask=mask)
    result = low_noise.create_low_noise_task.fn([good, bad])
    np.testing.assert_allclose(result.data, 1)
    np.testing.assert_allclose(result.uncertainty.array, 1)
    unsupported = low_noise.create_low_noise_task.fn([bad])
    assert np.all(np.isinf(unsupported.uncertainty.array))
    assert np.all(unsupported.mask)


@pytest.mark.parametrize("code, level, shape", [
    ("CTM", "3", (2, 2)), ("CTM", "Q", (2, 2)),
    ("PTM", "3", (3, 2, 2)), ("PTM", "Q", (3, 2, 2)),
])
def test_low_noise_shared_merger(sample_ndcube_clear, code, level, shape):
    """Clear, polarized and QuickPUNCH products all use the same RMS merger."""
    cubes = [sample_ndcube_clear(shape, code=code) for _ in range(2)]
    for cube, signal, sigma in zip(cubes, [1, 3], [1, 3]):
        cube.meta["LEVEL"] = level
        cube.data[:] = signal
        cube.uncertainty.array[:] = sigma
    expected = low_noise._merge_ndcubes(cubes)
    result = low_noise.create_low_noise_task.fn(cubes)
    np.testing.assert_allclose(result.data, 1.2)
    np.testing.assert_allclose(result.uncertainty.array, np.sqrt(5))
    np.testing.assert_array_equal(result.data, expected.data)
    np.testing.assert_array_equal(result.uncertainty.array, expected.uncertainty.array)
