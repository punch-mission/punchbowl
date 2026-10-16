import numpy as np
import pytest
from astropy.nddata import StdDevUncertainty

from punchbowl.data.tests.test_punch_io import sample_ndcube
from punchbowl.level3.polarization import convert_polarization


@pytest.mark.parametrize("errors", [(1, 1, 1), (1, 2, 3)])
def test_convert_polarization_uncertainty(sample_ndcube, errors):
    """Propagated variances must match the transform applied to basis images."""
    cube = sample_ndcube((3, 4, 4), code="PIM", level="3")
    sigma = np.broadcast_to(np.array(errors)[:, None, None] * 1e-14, cube.shape)
    cube = cube.replace(data=np.arange(48, dtype=float).reshape(cube.shape) / 48,
                        uncertainty=StdDevUncertainty(sigma))
    result = convert_polarization.fn(cube)

    variance = np.zeros(cube.shape)
    expected_data = np.zeros(cube.shape)
    for i in range(3):
        basis = np.zeros(cube.shape)
        basis[i] = 1
        # The signal transform supplies each Jacobian column independently.
        column = convert_polarization.fn(cube.replace(data=basis)).data
        variance += (column * sigma[i])**2
        expected_data += column * cube.data[i]
    np.testing.assert_allclose(result.uncertainty.array**2, variance, rtol=1e-12)
    np.testing.assert_allclose(result.data, expected_data, rtol=1e-12, atol=1e-15)
    assert result.meta.product_code == "PTM"
    if errors == (1, 1, 1):
        np.testing.assert_allclose(result.uncertainty.array[0]**2, 4 / 3 * 1e-28, rtol=1e-12)
        np.testing.assert_allclose(result.uncertainty.array[1:]**2, 8 / 3 * 1e-28, rtol=1e-12)


def test_convert_polarization_invalid_uncertainty(sample_ndcube):
    """A resolved triplet requires three valid input errors and unmasked samples."""
    cube = sample_ndcube((3, 4, 4), code="PIM", level="3")
    sigma = np.ones(cube.shape)
    sigma[0, 0] = [0, -1, np.nan, np.inf]
    mask = np.zeros(cube.shape, dtype=bool)
    mask[1, 1, 0] = True
    cube = cube.replace(uncertainty=StdDevUncertainty(sigma), mask=mask)
    result = convert_polarization.fn(cube)
    assert np.all(np.isinf(result.uncertainty.array[:, 0]))
    assert np.all(np.isinf(result.uncertainty.array[:, 1, 0]))
    assert np.all(result.mask[:, 0])
    assert np.all(result.mask[:, 1, 0])
    assert np.all(np.isfinite(result.uncertainty.array[:, 2:]))
    np.testing.assert_array_equal(cube.uncertainty.array, sigma)


def test_convert_polarization_without_uncertainty(sample_ndcube):
    """Preserve the signal and mask when input errors are unavailable."""
    cube = sample_ndcube((3, 4, 4), code="PIM", level="3")
    cube.uncertainty = None
    mask = np.zeros(cube.shape, dtype=bool)
    mask[0, 0, 0] = True
    result = convert_polarization.fn(cube.replace(mask=mask))
    assert result.uncertainty is None
    assert result.mask.shape == result.shape
    assert np.all(result.mask[:, 0, 0])
    assert np.all(np.isfinite(result.data))


def test_other_polarization_inputs_keep_original_errors(sample_ndcube):
    """Non-PIM conversion must retain its original missing-uncertainty behavior."""
    cube = sample_ndcube((3, 4, 4), code="PTM", level="3")
    result = convert_polarization.fn(cube)
    assert result.uncertainty is None
    assert result.mask is None
