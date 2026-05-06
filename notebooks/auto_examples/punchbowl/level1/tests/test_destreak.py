import pathlib

import numpy as np
import pytest
from pytest import fixture

from punchbowl.level1.destreak import correct_streaks, streak_correction_matrix

TEST_DIRECTORY = pathlib.Path(__file__).parent.resolve()


@pytest.mark.parametrize("diag, above, below", [(5, 4, 3), (5, 1, 1)])
def test_matrix_creation(diag, above, below):
    full_matrix = np.array(
        [
            [diag, above, above, above],
            [below, diag, above, above],
            [below, below, diag, above],
            [below, below, below, diag],
        ]
    )
    expected = np.linalg.inv(full_matrix)
    actual = streak_correction_matrix(4, diag, below, above)
    assert np.allclose(actual, expected)
    assert actual.shape == (4, 4)


@pytest.mark.parametrize(
    "size, exposure_time, readout_line_time, reset_line_time",
    [(5, 1, 2, 3), (100, 1, 2, 5)],
)
def test_blank_image_destreaking(
    size, exposure_time, readout_line_time, reset_line_time
):
    image = np.zeros((size, size))
    output = correct_streaks(image, exposure_time, readout_line_time, reset_line_time)
    assert output.shape == (size, size), "size should not change"
    assert np.allclose(output, 0), "an image of zeros destreaks to zeroes"


def test_not_square_image_destreaking():
    with pytest.raises(ValueError):
        image = np.zeros((50, 60))
        correct_streaks(image, 1, 2, 3)


def test_not_2dim_destreaking_errors():
    with pytest.raises(ValueError):
        image = np.zeros((50, 60, 70))
        correct_streaks(image, 1, 2, 3)


def test_not_numpy_image_destreaking_errors():
    with pytest.raises(TypeError):
        image = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        correct_streaks(image, 1, 2, 3)


def test_singular_matrix_errors():
    with pytest.raises(np.linalg.LinAlgError):
        image = np.zeros((3, 3))
        correct_streaks(image, 1, 1, 1)


# @pytest.mark.regression
# @fixture
# def used_correction_parameters() -> tuple:
#     exposure_time = 40
#     readout_line_time = 120 / 2148
#     reset_line_time = 120 / 2148
#     return exposure_time, readout_line_time, reset_line_time
#
#
# @pytest.mark.regression
# @fixture
# def expected_regression_test_input() -> np.ndarray:
#     """
#     This data is taken from the HotOp tests, specifically:
#         Nextcloud/23103_PUNCH_Data/WFI_PhaseC/PUNCH_WFI_EM_TBalance/archive_220116_HotOp_spot.fit
#     This is a calibrated 40 ms exposure following the steps in "Investigating destreaking WFI laboratory test data"
#     Jupyter notebook which can be found in the sdf at sdf/punchbowl/Level1/destreak_investigation_summary.ipynb
#
#     Returns
#     -------
#     np.ndarray
#         regression test input
#     """
#     path = TEST_DIRECTORY / "data/regression_input_destreak.npy"
#     return np.load(str(path))
#
#
# @pytest.mark.regression
# @fixture
# def expected_regression_test_output() -> np.ndarray:
#     path = TEST_DIRECTORY / "data/regression_output_destreak.npy"
#     return np.load(str(path))
#
#
# @pytest.mark.regression
# def test_regression(
#     expected_regression_test_input,
#     used_correction_parameters,
#     expected_regression_test_output,
# ):
#     test_output = correct_streaks(
#         expected_regression_test_input, *used_correction_parameters
#     )
#     assert np.allclose(test_output, expected_regression_test_output)
