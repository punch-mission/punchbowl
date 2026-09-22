import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from skimage.color import lab2rgb


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
    return LinearSegmentedColormap.from_list("PUNCH", rgb_colors, N=n)


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
    return LinearSegmentedColormap.from_list("PUNCH", rgb_colors, N=n)

cmap_punch_pb = _cmap_punch_pb()
cmap_punch_pb_r = _cmap_punch_pb().reversed()

cmap_punch = _cmap_punch()
cmap_punch_r = _cmap_punch().reversed()
