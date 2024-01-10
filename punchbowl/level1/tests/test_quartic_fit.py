import numpy as np
import pytest
from pytest import fixture

from punchbowl.level1.quartic_fit import (
    create_coefficient_image,
    create_constant_quartic_coefficients,
    photometric_calibration,
)


@fixture
def fake_one_image(shape=(10, 10)):
    """
    Parameters
    ----------
    shape : tuple
        the shape of the image to create

    Returns
    -------
    A fake image of the given shape with values of all 1

    """
    return np.ones(shape)


@fixture
def quartic_coefficients_image():
    """
    Returns
    -------
        Quartic coefficient image with flat_coefficients of [0, 1, 2, 3, 4]
    """
    flat_coeffs = np.array([0, 1, 2, 3, 4])
    image_shape = (10, 10)
    return create_coefficient_image(flat_coeffs, image_shape)


def test_create_constant_coefficient_image():
    image_shape = (10, 20)
    coeffs = create_constant_quartic_coefficients(image_shape)
    assert coeffs.shape[:-1] == image_shape
    assert coeffs.shape[-1] == 5
    for i in [0, 1, 2, 4]:
        assert np.all(coeffs[:, :, i] == 0)
    assert np.all(coeffs[:, :, 3] == 1)


def test_create_coeff_image_with_only_one_coeff():
    coeffs_image = create_coefficient_image(np.array([5]), (10, 10))
    assert coeffs_image.shape == (
        10,
        10,
        1,
    ), "Failed to make a coefficients image with only one coefficient"


def test_create_coeff_image_with_two_coeff():
    coeffs_image = create_coefficient_image(np.array([5, 6]), (10, 10))
    assert coeffs_image.shape == (
        10,
        10,
        2,
    ), "Failed to make a coefficients image with two coefficients"


def test_create_coefficient_image(quartic_coefficients_image):
    assert quartic_coefficients_image.shape == (
        10,
        10,
        5,
    ), "Quartic coefficient image shape is wrong."
    for i in range(quartic_coefficients_image.shape[-1]):
        assert np.all(
            quartic_coefficients_image[:, :, i] == i
        ), "Quartic coefficient image coefficients were not constructed properly."


@pytest.mark.parametrize("constant", [1, 2, 3])
def test_calibration_with_single_constant_coeff(
    fake_one_image, constant, shape=(10, 10)
):
    coeffs_image = create_coefficient_image(np.array([constant]), shape)
    calibrated = photometric_calibration(fake_one_image, coeffs_image)
    assert np.all(
        calibrated == constant
    ), "Photometric calibration with a single constant failed"


def test_calibration_with_all_one_coefficients(fake_one_image, shape=(10, 10)):
    coeffs_image = create_coefficient_image(np.array([1, 1, 1, 1, 1]), shape)
    calibrated = photometric_calibration(fake_one_image, coeffs_image)
    assert np.all(
        calibrated == 5
    ), "Photometric calibration with all one coefficients failed"


def test_calibration_with_range_coefficients(fake_one_image, shape=(10, 10)):
    coeffs_image = create_coefficient_image(np.array([1, 2, 3, 4, 5]), shape)
    calibrated = photometric_calibration(fake_one_image, coeffs_image)
    assert np.all(
        calibrated == 15
    ), "Photometric calibration with range coefficients failed"
