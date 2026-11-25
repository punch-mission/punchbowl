import numpy as np
import pytest
from astropy.wcs import WCS

from punchbowl.level1.pointing_refinement import (
    azp_model_from_wcs,
    check_projection_against_astropy,
    crota_to_pc,
    refine_pointing_matched,
)

NPIX = 2048


def make_wcs(crval=(130.0, 18.0), crota_deg=-23.0, mu=0.7, cdelt=0.0245):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "RA---AZP", "DEC--AZP"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.crpix = NPIX / 2 + 0.5, NPIX / 2 + 0.5
    wcs.wcs.cdelt = -cdelt, cdelt
    wcs.wcs.crval = crval
    wcs.wcs.pc = crota_to_pc(crota_deg)
    wcs.wcs.set_pv([(2, 1, mu)])
    wcs.wcs.set()
    return wcs


def sky_points(wcs, n, rng, margin=20):
    xy = rng.uniform(margin, NPIX - margin, (n, 2))
    ra, dec = wcs.pixel_to_world_values(xy[:, 0], xy[:, 1])
    return np.asarray(ra), np.asarray(dec), xy


def perturbed(wcs, dcrval=(0.15, -0.12), dcrota_deg=0.4):
    guess = wcs.deepcopy()
    guess.wcs.crval = wcs.wcs.crval + np.array(dcrval)
    guess.wcs.pc = crota_to_pc(azp_model_from_wcs(wcs).crota + dcrota_deg)
    guess.wcs.set()
    return guess


def max_field_error(wcs_a, wcs_b, rng):
    ra, dec, _ = sky_points(wcs_a, 2000, rng)
    ax, ay = wcs_a.world_to_pixel_values(ra, dec)
    bx, by = wcs_b.world_to_pixel_values(ra, dec)
    return np.max(np.hypot(ax - bx, ay - by))


@pytest.mark.parametrize("mu", [0.0, 0.7, 2.0])
@pytest.mark.parametrize("crval", [(130.0, 18.0), (359.9, -60.0), (10.0, 85.0)])
def test_fast_projection_matches_astropy(mu, crval):
    wcs = make_wcs(crval=crval, mu=mu)
    ra, dec, _ = sky_points(wcs, 2000, np.random.default_rng(0))
    assert check_projection_against_astropy(wcs, ra, dec) < 1e-6


def test_model_round_trips_wcs():
    wcs = make_wcs()
    model = azp_model_from_wcs(wcs)
    assert np.isclose(model.crota, -23.0)
    assert np.isclose(model.mu, 0.7)
    np.testing.assert_allclose(model.pc(), wcs.wcs.pc)


@pytest.mark.parametrize(("crval", "crota_deg"), [((130.0, 18.0), -23.0), ((359.95, 70.0), 179.8)])
def test_refine_recovers_pointing(crval, crota_deg):
    rng = np.random.default_rng(42)
    truth = make_wcs(crval=crval, crota_deg=crota_deg)
    ra, dec, xy = sky_points(truth, 300, rng)

    detected = rng.random(len(xy)) > 0.2
    observed = xy[detected] + rng.normal(0, 0.05, (detected.sum(), 2))
    observed = np.vstack([observed, rng.uniform(20, NPIX - 20, (40, 2))])  # spurious detections

    guess = perturbed(truth)
    assert max_field_error(truth, guess, rng) > 5

    refined, info = refine_pointing_matched(guess, ra, dec, observed)
    assert info.n_matched > 200
    assert info.rms_px < 0.1
    assert max_field_error(truth, refined, rng) < 0.05
    assert 0 <= refined.wcs.crval[0] < 360


@pytest.mark.filterwarnings("ignore:Only")
def test_refine_too_few_matches_raises():
    rng = np.random.default_rng(1)
    truth = make_wcs()
    ra, dec, _ = sky_points(truth, 50, rng)
    observed = rng.uniform(20, NPIX - 20, (3, 2))
    with pytest.raises(RuntimeError, match="too few"):
        refine_pointing_matched(truth, ra, dec, observed)


def test_refine_rejects_non_azp():
    wcs = make_wcs()
    wcs.wcs.ctype = "RA---TAN", "DEC--TAN"
    with pytest.raises(ValueError, match="AZP"):
        refine_pointing_matched(wcs, np.array([130.0]), np.array([18.0]), np.zeros((1, 2)))
