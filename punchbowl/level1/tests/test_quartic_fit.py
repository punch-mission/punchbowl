import numpy as np
import pytest
from pytest import fixture
from punchbowl.level1.quartic_fit import photometric_calibration


@fixture
def quartic_coeffs():
    # A synthetic set of quartic coefficients

    correction_coeffs = np.array([-8.82148244e-15,
                                  1.17884485e-09,
                                  -5.12648376e-05,
                                  1.87659792e+00,
                                  -4.79237873e+03])
    data_frame_dimensions = [10, 10]
    a_array = np.ones((data_frame_dimensions[0], data_frame_dimensions[1])) * correction_coeffs[0]
    b_array = np.ones((data_frame_dimensions[0], data_frame_dimensions[1])) * correction_coeffs[1]
    c_array = np.ones((data_frame_dimensions[0], data_frame_dimensions[1])) * correction_coeffs[2]
    d_array = np.ones((data_frame_dimensions[0], data_frame_dimensions[1])) * correction_coeffs[3]
    e_array = np.ones((data_frame_dimensions[0], data_frame_dimensions[1])) * correction_coeffs[4]
    CF_array = np.stack((a_array, b_array, c_array, d_array, e_array), axis=2)
    return CF_array


def test_sample_data_creation(quartic_coeffs):
    assert quartic_coeffs.shape[0] == 10


