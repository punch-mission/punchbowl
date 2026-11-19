from datetime import datetime

import numpy as np
import pytest
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from regularizepsf import ArrayPSF, simple_functional_psf
from regularizepsf.util import calculate_covering

from punchbowl.data import NormalizedMetadata
from punchbowl.data.punchcube import PUNCHCube
from punchbowl.level1.dynamic_psf import correct_dynamic_psf_task

IMAGE_SIZE = 512
PSF_SIZE = 32
BOX = (100, 400, 100, 400)


def make_target_psf(sigma: float = 1.5) -> ArrayPSF:
    center = PSF_SIZE / 2

    @simple_functional_psf
    def target(row, col, x0=center, y0=center, sigma_x=sigma, sigma_y=sigma):
        return np.exp(-(np.square(row - x0) / (2 * np.square(sigma_x))
                        + np.square(col - y0) / (2 * np.square(sigma_y))))

    coords = calculate_covering((IMAGE_SIZE, IMAGE_SIZE), PSF_SIZE)
    return target.as_array_psf(coords, PSF_SIZE)


def make_star_field(sigma: float = 2.5, seed: int = 0) -> PUNCHCube:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:IMAGE_SIZE, :IMAGE_SIZE]
    data = np.full((IMAGE_SIZE, IMAGE_SIZE), 10.0)
    for y, x in rng.uniform(BOX[0] + PSF_SIZE, BOX[1] - PSF_SIZE, size=(40, 2)):
        data += 1000 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2))
    data += rng.normal(0, 0.5, data.shape)

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.02, 0.02
    wcs.wcs.crpix = IMAGE_SIZE / 2, IMAGE_SIZE / 2
    wcs.wcs.crval = 0, 0

    meta = NormalizedMetadata.load_template("PM1", "1")
    meta["DATE-OBS"] = str(datetime(2024, 2, 22, 16, 0, 1))
    return PUNCHCube(data=data, uncertainty=StdDevUncertainty(np.zeros_like(data)), wcs=wcs, meta=meta)


def test_correct_dynamic_psf_task_skips_without_target():
    cube = make_star_field()
    original = cube.data.copy()

    result = correct_dynamic_psf_task.fn(cube, None)

    assert np.array_equal(result.data, original)
    assert "no dynamic PSF correction applied" in result.meta.history[-1].comment


@pytest.mark.parametrize("use_path_string", [True, False])
def test_correct_dynamic_psf_task_applies_saved_target(tmp_path, use_path_string):
    target_path = tmp_path / "PUNCH_L1_DFM_20250101000000_v1.fits"
    make_target_psf().save(target_path)
    cube = make_star_field()
    original = cube.data.copy()

    target = str(target_path) if use_path_string else target_path
    with pytest.MonkeyPatch.context() as mp:
        # use a box and PSF size suited to the small synthetic image
        from punchbowl.level1 import dynamic_psf
        original_correct = dynamic_psf.correct_psf_per_image
        mp.setattr(dynamic_psf, "correct_psf_per_image",
                   lambda c, t: original_correct(c, t, psf_size=PSF_SIZE, box=BOX))
        result = correct_dynamic_psf_task.fn(cube, target)

    assert result.data.shape == original.shape
    assert np.all(np.isfinite(result.data))
    assert not np.array_equal(result.data, original)
    assert target_path.name in result.meta.history[-1].comment
