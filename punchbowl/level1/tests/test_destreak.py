import numpy as np
import pytest
from punchbowl.level1.destreak import correct_streaks, streak_correction_matrix


@pytest.mark.parametrize("diag, above, below",
                         [(5, 4, 3),
                          (5, 1, 1)])
def test_matrix_creation(diag, above, below):
    M = np.array([[diag, above, above, above],
                 [below, diag, above, above],
                 [below, below, diag, above],
                 [below, below, below, diag]])
    expected = np.linalg.inv(M)
    actual = streak_correction_matrix(4, diag, below, above)
    assert np.allclose(actual, expected)
    assert actual.shape == (4, 4)


def test_actual_destreaking():
    # TODO: make this into a more robust test instead of just the shape test
    size = 500
    image = np.zeros((size, size))
    exposure_time = 100
    readout_line_time = 5
    reset_line_time = 5
    output = correct_streaks(image, exposure_time, readout_line_time, reset_line_time)
    assert output.shape == (size, size)

