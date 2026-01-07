import numpy as np
import matplotlib.colors as mcolors
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

cmap_punch = _cmap_punch()
cmap_punch_r = _cmap_punch().reversed()


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

def generate_mzp_to_rgb_map(data_cube: np.ndarray,
                            gamma:float=0.7,
                            frac:float=0.125,
                            s_boost:float=2.25) -> np.ndarray:
    """
    Create an RGB composite from a MZP cube.

    Parameters
    ----------
    data_cube : NDData-like or numpy array
        Expected shape: (3, ny, nx)
        Channels correspond to M, Z, P images.
    gamma : float
        Power-law exponent to apply to each channel.
    frac : float
        Fractional scaling applied after median normalization.
    s_boost : float
        HSV saturation boost factor (>1 increases color saturation).

    Returns
    -------
    rgb_sat : ndarray (ny, nx, 3)
        Float RGB array in [0,1] with enhanced saturation.
    color_image : ndarray (3, ny, nx)
        8-bit RGB image before HSV saturation.

    """
    m = data_cube[0].astype(np.float32)
    z = data_cube[1].astype(np.float32)
    p = data_cube[2].astype(np.float32)

    m = m ** gamma
    z = z ** gamma
    p = p ** gamma

    # Median-normalize and scale to 0 to 255 range
    scaled_m = (np.clip(frac * m / np.nanmedian(m), 0, 1) * 255).astype("float32")
    scaled_z = (np.clip(frac * z / np.nanmedian(z), 0, 1) * 255).astype("float32")
    scaled_p = (np.clip(frac * p / np.nanmedian(p), 0, 1) * 255).astype("float32")

    ny, nx = m.shape
    color_image = np.zeros((3, ny, nx), dtype=np.uint16)
    color_image[0] = scaled_m
    color_image[1] = scaled_z
    color_image[2] = scaled_p

    # Convert to RGB (ny, nx, 3)
    rgb = np.moveaxis(color_image, 0, -1) / 255.0

    # RGB to HSV
    hsv = mcolors.rgb_to_hsv(rgb)

    # Boost saturation
    hsv[..., 1] = np.clip(hsv[..., 1] * s_boost, 0, 1)

    # HSV to RGB
    rgb_sat = mcolors.hsv_to_rgb(hsv)

    return rgb_sat, color_image

