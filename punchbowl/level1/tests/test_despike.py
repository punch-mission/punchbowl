import numpy as np

from punchbowl.level1.despike import radial_array, spikejones, astroscrappy_despike


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
    assert np.isclose(output[50, 50], 0.1)
    assert np.isclose(output[49, 50], 0.1)
    assert np.isclose(output[30, 70], 0.1)


def test_astroscrappy_with_one_spike():
    image = np.zeros((100, 100)) + 0.1
    image[50, 50] = 100
    image[49, 50] = 10

    image[30, 70] = 5
    output, spikes = astroscrappy_despike(image,
                            sigclip=4.5, sigfrac=0.3,
                            objlim=5.0, gain=1.0, readnoise=0)

    assert output.shape == image.shape
    assert np.isclose(spikes[50, 50], 1)
    assert np.isclose(spikes[49, 50], 1)
    assert np.isclose(spikes[30, 70], 1)