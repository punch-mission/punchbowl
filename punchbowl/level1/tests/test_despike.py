import numpy as np

from punchbowl.level1.despike import radial_array, spikejones


def test_create_radial_array():
    shape = (7, 7)
    center = (2, 2)
    arr = radial_array(shape, center=center)
    expected = np.array([
             [2, 2, 2, 2, 2, 3, 4],
             [2, 1, 1, 1, 2, 3, 4],
             [2, 1, 0, 1, 2, 3, 4],
             [2, 1, 1, 1, 2, 3, 4],
             [2, 2, 2, 2, 2, 3, 4],
             [3, 3, 3, 3, 3, 4, 5],
             [4, 4, 4, 4, 4, 5, 5]
            ])
    assert isinstance(arr, np.ndarray)
    assert np.all(arr == expected)


def test_spikejones_with_one_spike():
    image = np.zeros((100, 100)) + 0.1
    image[50, 50] = 100
    image[49, 50] = 10

    image[30, 70] = 5
    output, spikes = spikejones(image)

    assert output.shape == image.shape
    assert output[50, 50]
    assert output[49, 50]
    assert output[30, 70]
