import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from punchbowl.data.visualize import cmap_punch, radial_distance, radial_filter


def test_radial_distance():
    shape = (100, 100)
    radial_array = radial_distance(shape[0], shape[1])
    assert isinstance(radial_array, np.ndarray)
    assert radial_array.shape == shape


def test_radial_filter():
    shape = (100, 100)
    data_array = np.random.random(shape)
    filtered_array = radial_filter(data_array)
    assert isinstance(filtered_array, np.ndarray)
    assert filtered_array.shape == shape


def test_cmap_punch():
    cmap = cmap_punch()
    assert isinstance(cmap, LinearSegmentedColormap)
