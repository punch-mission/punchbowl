import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from punchbowl.data.tests.test_punch_io import sample_ndcube
from punchbowl.data.visualize import (
    cmap_punch,
    cmap_punch_r,
    generate_tbpb_to_rgb_map,
    plot_punch,
    radial_distance,
    radial_filter,
)


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
    assert isinstance(cmap_punch, LinearSegmentedColormap)


def test_cmap_punch_r():
    assert isinstance(cmap_punch_r, LinearSegmentedColormap)


def test_plot_punch(sample_ndcube):
    cube = sample_ndcube(shape=(10, 10), code="CAM", level="3")

    fig, ax = plot_punch(cube)

    assert fig is not None
    assert ax is not None

    plt.close(fig)


def test_plot_punch_options(sample_ndcube):
    cube = sample_ndcube(shape=(10, 10), code="CAM", level="3")

    fig, ax = plot_punch(cube, trim_edge=(0.1, 0.9), axes_off=True, vmin=0, vmax=1)

    assert fig is not None
    assert ax is not None

    plt.close(fig)


def test_tbpb_to_rgb(sample_ndcube):
    cube = sample_ndcube((2, 16, 16), code="PTM", level="3")
    rgb, color = generate_tbpb_to_rgb_map(cube)
    assert rgb.shape == (16, 16, 3)
    assert color.shape == (3, 16, 16)
    assert 0 <= rgb.min() and rgb.max() <= 1
