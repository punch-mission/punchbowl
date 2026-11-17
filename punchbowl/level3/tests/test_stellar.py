import os
from datetime import timedelta

import numpy as np
import pytest
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from reproject import reproject_adaptive

from punchbowl.data import NormalizedMetadata, punch_io
from punchbowl.data.punchcube import PUNCHCube
from punchbowl.level3 import stellar
from punchbowl.level3.stellar import (
    polarize_celestial_to_solar,
    polarize_solar_to_celestial,
    subtract_starfield_background_task,
)

TESTDATA_DIR = os.path.dirname(__file__)
TEST_FILE = TESTDATA_DIR + '/data/downsampled_L2_PTM.fits'

def test_solar_celestial():
    data_cube = punch_io.load_ndcube_from_fits(TEST_FILE, include_provenance=False)
    data_cel = polarize_solar_to_celestial(data_cube)
    data_solar = polarize_celestial_to_solar(data_cel)

    assert isinstance(data_cel, PUNCHCube)
    assert isinstance(data_solar, PUNCHCube)
    assert np.allclose(data_solar.data, data_cube.data, atol=1e-30)
    assert data_cel.data.shape == data_cube.data.shape
    assert not np.shares_memory(data_cube.data, data_cel.data)
    assert not np.shares_memory(data_solar.data, data_cel.data)


STAR_AMPLITUDE = 100.0
BACKGROUND = 5.0
MODEL_UNCERTAINTY = 0.5
IMAGE_UNCERTAINTY = 2.0


def _celestial_wcs(scale: float, n: int, crval: tuple[float, float] = (30.0, 10.0)) -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "RA---ARC", "DEC--ARC"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = -scale, scale
    wcs.wcs.crpix = n / 2 + 0.5, n / 2 + 0.5
    wcs.wcs.crval = crval
    wcs.array_shape = (n, n)
    return wcs


def _star_image(shape: tuple[int, int], n_stars: int = 30, margin: int = 20, seed: int = 0) \
        -> tuple[np.ndarray, np.ndarray]:
    """Make an image of PSF-sized (sigma = 1 px) stars on a flat background."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    stars = rng.uniform(margin, np.array(shape) - margin, size=(n_stars, 2))
    image = np.full(shape, BACKGROUND)
    for y, x in stars:
        image += STAR_AMPLITUDE * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 2)
    return image, np.round(stars).astype(int)


def _build_model(image: np.ndarray, image_wcs: WCS, map_scale: float) -> tuple[np.ndarray, WCS]:
    """Reproject an image into a starfield map the same way remove_starfield builds its estimates."""
    n_map = int(image.shape[-1] * abs(image_wcs.wcs.cdelt[0]) * 1.1 / map_scale)
    map_wcs = _celestial_wcs(map_scale, n_map, crval=tuple(image_wcs.wcs.crval[:2]))
    model = reproject_adaptive((image, image_wcs), map_wcs, map_wcs.array_shape, return_footprint=False,
                               roundtrip_coords=False, boundary_mode="strict", conserve_flux=True)
    return model, map_wcs


def _model_cube(data: np.ndarray, wcs: WCS, date_obs: str, typecode: str, image_scale: float) -> PUNCHCube:
    meta = NormalizedMetadata.load_template(typecode, "3")
    meta["DATE-OBS"] = date_obs
    # Like the model data, the model uncertainty is in per-map-pixel units, so that it's MODEL_UNCERTAINTY once
    # flux-conservingly reprojected into the image frame
    uncertainty = MODEL_UNCERTAINTY * (abs(wcs.wcs.cdelt[0]) / image_scale) ** 2
    return PUNCHCube(data=data, uncertainty=StdDevUncertainty(np.full_like(data, uncertainty)),
                     wcs=wcs, celestial_wcs=wcs, meta=meta)


def _patch_model_loading(monkeypatch: pytest.MonkeyPatch, models: dict[str, PUNCHCube]) -> None:
    monkeypatch.setattr(stellar, "load_ndcube_from_fits", lambda path, *args, **kwargs: models[path])


def _clear_scene(model_scale: float = 1.0, after_scale: float = 1.0):
    """A clear image of stars, plus before/after starfield models built from it on a finer 0.01 deg map."""
    image_wcs = _celestial_wcs(0.0225, 200)
    image, stars = _star_image(image_wcs.array_shape)

    meta = NormalizedMetadata.load_template("CTM", "3")
    meta["DATE-OBS"] = "2025-06-01T12:00:00.000"
    data_object = PUNCHCube(data=image.copy(), uncertainty=StdDevUncertainty(np.full_like(image, IMAGE_UNCERTAINTY)),
                            wcs=image_wcs, celestial_wcs=image_wcs, meta=meta)

    model, map_wcs = _build_model(image, image_wcs, map_scale=0.01)
    models = {"before": _model_cube(model * model_scale, map_wcs, "2025-06-01T06:00:00.000", "CSM", 0.0225),
              "after": _model_cube(model * after_scale, map_wcs, "2025-06-02T06:00:00.000", "CSM", 0.0225)}
    return data_object, models, stars


def _star_peaks(data: np.ndarray, stars: np.ndarray) -> np.ndarray:
    return np.array([data[..., y, x] for y, x in stars]).T


def test_subtract_starfield_removes_stars(monkeypatch):
    """Regression test: stars were previously left at ~30-80% after subtraction, depending on the map scale."""
    data_object, models, stars = _clear_scene()
    _patch_model_loading(monkeypatch, models)

    output = subtract_starfield_background_task.fn(data_object, "before", "after", is_polarized=False)

    interior = output.data[20:-20, 20:-20]
    assert np.all(np.abs(_star_peaks(output.data, stars)) < 0.02 * STAR_AMPLITUDE)
    assert np.abs(interior).max() < 0.02 * STAR_AMPLITUDE
    # The diffuse background must be removed too, not just the stars
    assert np.abs(np.median(interior)) < 0.01 * BACKGROUND


def test_subtract_starfield_interpolates_models_in_time(monkeypatch):
    """The observation is 1/4 of the way from the before model to the after model, so 1.25x the stars are removed."""
    data_object, models, stars = _clear_scene(model_scale=1.0, after_scale=2.0)
    _patch_model_loading(monkeypatch, models)

    output = subtract_starfield_background_task.fn(data_object, "before", "after", is_polarized=False)

    # The residual should be -1/4 of the image (as blurred by remove_starfield) everywhere
    background_residual = np.median(output.data[20:-20, 20:-20])
    assert background_residual == pytest.approx(-0.25 * BACKGROUND, rel=0.02)
    peaks = _star_peaks(output.data, stars) - background_residual
    assert np.all(peaks < 0)
    assert np.all(peaks > -0.25 * STAR_AMPLITUDE)


def test_subtract_starfield_combines_uncertainties(monkeypatch):
    data_object, models, _ = _clear_scene()
    _patch_model_loading(monkeypatch, models)

    output = subtract_starfield_background_task.fn(data_object, "before", "after", is_polarized=False)

    expected = np.sqrt(IMAGE_UNCERTAINTY**2 + MODEL_UNCERTAINTY**2)
    assert np.allclose(output.uncertainty.array[20:-20, 20:-20], expected, rtol=0.01)


def test_subtract_starfield_preserves_mask(monkeypatch):
    data_object, models, _ = _clear_scene()
    data_object.data[:, :50] = 0
    data_object.uncertainty.array[:, :50] = np.inf
    _patch_model_loading(monkeypatch, models)

    output = subtract_starfield_background_task.fn(data_object, "before", "after", is_polarized=False)

    assert np.all(output.data[:, :50] == 0)
    assert np.all(np.isfinite(output.data))


def test_subtract_starfield_polarized(monkeypatch):
    """Polarized stars are removed, which requires subtracting in the celestial polarization frame."""
    sample = punch_io.load_ndcube_from_fits(TEST_FILE, include_provenance=False)
    celestial_wcs = sample.celestial_wcs
    celestial_wcs_2d = celestial_wcs.celestial

    # Stars with different brightnesses in each celestial-frame polarizer channel, as the starfield model sees them
    star_image, stars = _star_image(sample.data.shape[-2:], n_stars=40, margin=200, seed=1)
    celestial_data = np.stack([star_image * f for f in (1.0, 0.6, 0.3)])
    # ...and as the observatory sees them, in the mzpsolar frame
    observed = polarize_celestial_to_solar(
        PUNCHCube(data=celestial_data, wcs=celestial_wcs, meta=sample.meta)).data
    # The frame rotation must actually change the data, or this test can't detect a frame mismatch
    assert not np.allclose(observed, celestial_data, rtol=0.05)

    data_object = PUNCHCube(data=observed.copy(),
                            uncertainty=StdDevUncertainty(np.full_like(observed, IMAGE_UNCERTAINTY)),
                            wcs=sample.wcs, celestial_wcs=celestial_wcs, meta=sample.meta)

    model, map_wcs = _build_model(celestial_data, celestial_wcs_2d, map_scale=0.08)
    before = (sample.meta.datetime - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%S.000")
    after = (sample.meta.datetime + timedelta(hours=18)).strftime("%Y-%m-%dT%H:%M:%S.000")
    image_scale = abs(celestial_wcs_2d.wcs.cdelt[0])
    _patch_model_loading(monkeypatch, {"before": _model_cube(model, map_wcs, before, "PSM", image_scale),
                                       "after": _model_cube(model, map_wcs, after, "PSM", image_scale)})

    output = subtract_starfield_background_task.fn(data_object, "before", "after", is_polarized=True)

    assert output.data.shape == observed.shape
    assert np.all(np.abs(_star_peaks(output.data, stars)) < 0.02 * STAR_AMPLITUDE)
    assert np.abs(output.data[:, 200:-200, 200:-200]).max() < 0.02 * STAR_AMPLITUDE
