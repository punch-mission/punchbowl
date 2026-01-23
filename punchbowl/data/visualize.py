from pathlib import Path

import astropy.units as u
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize, PowerNorm
from matplotlib.figure import Figure
from ndcube import NDCube
from skimage.color import lab2rgb

from punchbowl.data import punch_io


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


def plot_punch(
    data: Path | NDCube,
    layer: int = 0,
    cmap: str | Colormap | None = cmap_punch,
    norm: Normalize | None = PowerNorm,
    vmin: float = 1e-14,
    vmax: float = 1e-12,
    gamma: float = 1/2.2,
    figsize: tuple[float, float] = (10,8),
    axes_labels: tuple[str, str] = ("Helioprojective longitude", "Helioprojective latitude"),
    axes_off: bool = False,
    annotate: bool = True,
    grid_spacing: int = 15,
    grid_alpha: float = 0.25,
    title_prefix: str | None = None,
    colorbar: bool = True,
    colorbar_label: str = "Mean Solar Brightness (MSB)",
    save_path: str | Path | None = None,
    dpi: int = 300,
    ) -> tuple[Figure, Axes]:
    """
    Plot a PUNCH NDCube data object.

    Parameters
    ----------
    data : Path | NDCube
        PUNCH data to plot, either a filepath or an NDCube
    layer : int
        Data layer to plot when using three-dimensional data cubes
    cmap : str or Colormap, optional
        Colormap to use for plot
    norm : Normalize, optional
        Normalization function for image
    vmin : float, optional
        Normalization vmin value
    vmax : float, optional
        Normalization vmax value
    gamma : float, optional
        Normalization gamma scaling value
    figsize : tuple, optional
        Figure size
    axes_labels : tuple[str, str], optional
        Axes labels (x, y)
    axes_off : bool, optional
        Remove axes and labels
    annotate : bool, optional
        Toggles display of corner annotation when axes_off is True
    grid_spacing : int, optional
        Coordinate grid spacing in degrees, removes grid for None
    grid_alpha : float, optional
        Coordinate grid transparency (1: opaque, 0: transparent)
    title_prefix : str, optional
        Prefix to prepend to plot title
    colorbar : bool, optional
        Toggle for plotting colorbar
    colorbar_label : str, optional
        Label to use for the colorbar
    save_path : str or Path, optional
        When provided, saves the figure to file directly without plotting on screen
    dpi : int, optional
        DPI for output plots saved to file

    Returns
    -------
    tuple of (figure, axes)

    """
    if isinstance(data, NDCube):
        cube = data
    elif isinstance(data, Path | str):
        cube = punch_io.load_ndcube_from_fits(data)
    else:
        msg = "Provide a valid file path or NDCube for plotting."
        raise TypeError(msg)

    norm = norm(gamma, vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": cube.wcs})

    im = ax.imshow(cube.data if cube.data.ndim == 2 else cube.data[layer,...], cmap=cmap, norm=norm)

    lon, lat = ax.coords
    lat.set_ticks(np.arange(-90, 90, grid_spacing) * u.degree)
    lon.set_ticks(np.arange(-180, 180, grid_spacing) * u.degree)
    lat.set_major_formatter("dd")
    lon.set_major_formatter("dd")

    ax.set_facecolor("black")
    ax.coords.grid(color="white", alpha=grid_alpha, ls="dotted")
    ax.set_xlabel(axes_labels[0])
    ax.set_ylabel(axes_labels[1])
    timestamp = cube.meta.datetime.strftime("%Y/%m/%d %H:%M:%S UT")

    if title_prefix is None:
        title_prefix = f"PUNCH {cube.meta['TYPECODE']}{cube.meta['OBSCODE']}"
    ax.set_title(f"{title_prefix} {timestamp}")

    if axes_off:
        ax.set_axis_off()
        ax.set_title("")
        fig.set_facecolor("black")

    if axes_off and annotate:
        ax.text(0.01,0.01,
        cube.meta.datetime.strftime("%Y-%m-%d %H:%M:%S UT"),
        transform=ax.transAxes,
        color="white",
        verticalalignment="bottom",
        horizontalalignment="left",
        fontsize=8,
        fontfamily="monospace")

        ax.text(0.01, 0.05,
        title_prefix,
        transform=ax.transAxes,
        color="white",
        verticalalignment="bottom",
        horizontalalignment="left",
        fontsize=8,
        fontfamily="monospace")

    if colorbar and not axes_off:
        fig.colorbar(im, ax=ax, label=colorbar_label)

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return None

    return fig, ax
