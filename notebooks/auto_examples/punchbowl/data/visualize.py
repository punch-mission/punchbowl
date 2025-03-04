import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from skimage.color import lab2rgb


def cmap_punch() -> LinearSegmentedColormap:
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


def radial_distance(h: int, w: int, center: tuple[int, int] | None = None, radius: float | None = None) -> np.ndarray:
    """Create radial distance array."""
    if center is None:
        center = (int(w/2), int(h/2))

    if radius is None:
        radius = min([center[0], center[1], w-center[0], h-center[1]])

    y, x = np.ogrid[:h, :w]
    dist_arr = np.sqrt((x - center[0])**2 + (y - center[1])**2)

    return dist_arr / dist_arr.max()


def radial_filter(data: np.ndarray) -> np.ndarray:
    """Filter data with radial distance function."""
    return data * radial_distance(*data.shape) ** 2.5
