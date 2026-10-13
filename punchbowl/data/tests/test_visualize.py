from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import LinearSegmentedColormap

from punchbowl.data.tests.test_punch_io import sample_ndcube
from punchbowl.data.visualize import (
    DEFAULT_SCALING,
    animate_punch,
    cmap_punch,
    cmap_punch_r,
    generate_mzp_to_rgb_map,
    load_quicklook_scaling,
    plot_punch,
    radial_distance,
    radial_filter,
)

CONFIG_SCALING_PATH = Path(__file__).resolve().parent.parent / "data" / "scaling_parameters.yaml"

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


def test_load_quicklook_scaling():
    vmin, vmax = load_quicklook_scaling(level="0", product="CR", obscode="2", config=str(CONFIG_SCALING_PATH))

    assert vmin == 100
    assert vmax == 800


def test_load_quicklook_scaling_no_input():
    vmin, vmax = load_quicklook_scaling(config=str(CONFIG_SCALING_PATH))

    assert vmin == 5e-13
    assert vmax == 5e-11


def test_load_quicklook_scaling_no_product():
    vmin, vmax = load_quicklook_scaling(level="0", config=str(CONFIG_SCALING_PATH))

    assert vmin == 100
    assert vmax == 800


def test_radial_distance():
    r = radial_distance(100, 100)
    assert np.allclose(r.max(), 1)
    assert r[50, 50] == 0


def test_generate_mzp_to_rgb_map():
    data_cube = np.random.random((3, 16, 16))
    rgb, color_image = generate_mzp_to_rgb_map(data_cube)
    assert rgb.shape == (16, 16, 3)
    assert color_image.shape == (3, 16, 16)
    assert rgb.min() >= 0
    assert rgb.max() <= 1


def test_load_quicklook_scaling():
    vmin, vmax = load_quicklook_scaling(level="0", product="CR", obscode="3", config=str(CONFIG_SCALING_PATH))
    assert vmin < vmax


def test_load_quicklook_scaling_dict_config():
    config = {"quicklook_scaling": {"0": {"CR": [[1, 2], [3, 4]], "default": [[5, 6], [7, 8]]}}}
    assert tuple(load_quicklook_scaling("0", "CR", "2", config)) == (1, 2)
    assert tuple(load_quicklook_scaling("0", "CR", "4", config)) == (3, 4)
    assert tuple(load_quicklook_scaling("0", "??", "2", config)) == (5, 6)


def test_load_quicklook_scaling_none():
    vmin, vmax = load_quicklook_scaling(level="0", product="CR", obscode="3", config=None)
    assert vmin < vmax


def test_load_quicklook_scaling_missing_key():
    assert load_quicklook_scaling("0", "CR", "2", config={}) == DEFAULT_SCALING


def test_load_quicklook_scaling_bad_config_type():
    with pytest.raises(TypeError):
        load_quicklook_scaling(config=123)


def test_plot_punch_bad_input():
    with pytest.raises(TypeError):
        plot_punch(42)


def test_plot_punch_no_set_scaling(sample_ndcube):
    cube = sample_ndcube(shape=(10, 10), code="CTM", level="3")
    cube.meta["LEVEL"] = "-1"
    with pytest.raises(KeyError):
        fig, ax = plot_punch(cube)


def test_plot_punch_save_path(sample_ndcube, tmp_path):
    cube = sample_ndcube(shape=(10, 10), code="CAM", level="3")
    out = tmp_path / "out.png"
    assert plot_punch(cube, save_path=out, dpi=50) is None
    assert out.exists()


def test_plot_punch_title_prefix(sample_ndcube):
    cube = sample_ndcube(shape=(10, 10), code="CAM", level="3")
    fig, ax = plot_punch(cube, title_prefix="TEST")
    assert ax.get_title().startswith("TEST")
    plt.close(fig)


def test_plot_punch_labels(sample_ndcube):
    cube = sample_ndcube(shape=(10, 10), code="CTM", level="3")
    for n in (1, 2, 3):
        cube.meta[f"CTRXWFI{n}"] = 2 * n
        cube.meta[f"CTRYWFI{n}"] = 3 * n
    fig, ax = plot_punch(cube, label_satellites=True, legend_satellites=True)
    assert {"1", "2", "3"} <= {t.get_text() for t in ax.texts}
    assert {"WFI1", "WFI2", "WFI3"} <= {t.get_text() for t in ax.texts}
    plt.close(fig)


def test_plot_punch_float_trim_no_colorbar(sample_ndcube):
    cube = sample_ndcube(shape=(10, 10), code="CAM", level="3")
    fig, _ = plot_punch(cube, trim_edge=0.5, colorbar=False, label_satellites=True)
    assert len(fig.axes) == 1  # no colorbar axis
    plt.close(fig)


def test_plot_punch_axes_off_colorbar(sample_ndcube):
    cube = sample_ndcube(shape=(10, 10), code="CAM", level="3")
    fig, ax = plot_punch(cube, axes_off=True, colorbar=True, vmin=1e-14, vmax=1e-12)
    assert len(ax.child_axes) == 1
    plt.close(fig)
