from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as colors
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from skimage.color import lab2rgb

_CMAP_PATH = Path(__file__).parent / "data"

# Color table csv file loading and registration adapted from SunPy
# (https://github.com/sunpy/sunpy/blob/main/sunpy/visualization/colormaps/color_tables.py)
def create_cdict(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> dict:
    """Create the color tuples in the correct format."""
    i = np.linspace(0, 1, r.size)
    return {name: list(zip(i, el / 255.0, el / 255.0, strict=False))
             for el, name in [(r, "red"), (g, "green"), (b, "blue")]}


def _cmap_from_rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray, name: str) -> LinearSegmentedColormap:
    cdict = create_cdict(r, g, b)
    return colors.LinearSegmentedColormap(name, cdict)


def cmap_from_rgb_file(name: str, fname: str) -> LinearSegmentedColormap:
    """Create a colormap from a RGB .csv file."""
    data = np.loadtxt(_CMAP_PATH / fname, delimiter=",")
    if data.shape[1] != 3:
        raise RuntimeError(f"RGB data files must have 3 columns (got {data.shape[1]})")
    return _cmap_from_rgb(data[:, 0], data[:, 1], data[:, 2], name)


# PUNCH total brightness
def _cmap_punch() -> LinearSegmentedColormap:
    """Generate PUNCH colormap."""
    # Define key colors in LAB space
    black_lab = np.array([0, 0, 0])
    orange_lab = np.array([50, 15, 50])
    white_lab = np.array([100, 0, 0])

    # Define the number of colors
    n = 256
    lab_colors = np.zeros((n, 3))

    # Transition from black to orange
    for i in range(n // 2):
        t = i / (n // 2 - 1)
        lab_colors[i] = black_lab * (1 - t) + orange_lab * t

    # Transition from orange to white
    for i in range(n // 2, n):
        t = (i - n // 2) / (n // 2 - 1)
        lab_colors[i] = orange_lab * (1 - t) + white_lab * t

    rgb_colors = lab2rgb(lab_colors.reshape(1, -1, 3)).reshape(n, 3)
    return LinearSegmentedColormap.from_list("PUNCH_tB", rgb_colors, N=n)

cmap_punch = _cmap_punch()
cmap_punch_r = _cmap_punch().reversed()


# PUNCH polarized brightness
def _cmap_punch_pb() -> LinearSegmentedColormap:
    """Generate PUNCH colormap."""
    black_lab = np.array([0, 0, 0])
    midpoint_lab = np.array([50, 40, 90])
    white_lab = np.array([100, 0, 0])

    n = 256
    lab_colors = np.zeros((n, 3))

    # black to midpoint
    for i in range(n // 2):
        t = i / (n // 2 - 1)
        lab_colors[i] = black_lab * (1 - t) + midpoint_lab * t

    # midpoint to white
    for i in range(n // 2, n):
        t = (i - n // 2) / (n // 2 - 1)
        lab_colors[i] = midpoint_lab * (1 - t) + white_lab * t

    rgb_colors = lab2rgb(lab_colors.reshape(1, -1, 3)).reshape(n, 3)
    return LinearSegmentedColormap.from_list("PUNCH_pB", rgb_colors, N=n)

cmap_punch_pb = _cmap_punch_pb()
cmap_punch_pb_r = _cmap_punch_pb().reversed()


# PUNCH tau
def _cmap_punch_tau() -> LinearSegmentedColormap:
    return cmap_from_rgb_file("punch_tau", "cmap_punch_tau.csv")

cmap_punch_tau = _cmap_punch_tau()
cmap_punch_tau_r = _cmap_punch_tau().reversed()


# PUNCH p
cmap_punch_p = mpl.colormaps["plasma"]
cmap_punch_p_r = mpl.colormaps["plasma"].reversed()


colormap_list = {
    "punch_r": cmap_punch_r,
    "punch_tb": cmap_punch,
    "punch_tb_r": cmap_punch_r,
    "punch_pb": cmap_punch_pb,
    "punch_pb_r": cmap_punch_pb_r,
    "punch_tau": cmap_punch_tau,
    "punch_tau_r": cmap_punch_tau_r,
    "punch_p": cmap_punch_p,
    "punch_p_r": cmap_punch_p_r,
}

for name, cmap in colormap_list.items():
    mpl.colormaps.register(cmap, name=name)
