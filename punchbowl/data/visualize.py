import matplotlib.colors as mcolors
import numpy as np
from astropy.io import fits
from matplotlib.colors import LogNorm
from openjpeg.utils import encode_array
from skimage.color import lab2rgb


def cmap_punch():
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
    return mcolors.LinearSegmentedColormap.from_list("PUNCH", rgb_colors, N=n)

def radial_distance(h, w, center=None, radius=None):
    if center is None:
        center = (int(w/2), int(h/2))

    if radius is None:
        radius = min([center[0], center[1], w-center[0], h-center[1]])

    Y, X = np.ogrid[:h, :w]
    dist_arr = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    return dist_arr / dist_arr.max()

def radial_filter(data):
    return data * radial_distance(*data.shape) ** 2.5


if __name__ == "__main__":
    cmap = cmap_punch()
    norm = LogNorm(vmin=0, vmax=8_000)
    # norm = Normalize(vmin=1E-15, vmax=8e-13)

    path = "/Users/jhughes/Desktop/data/punch build 3 synthetic data/d0/2/PTM/2024/08/25/PUNCH_L2_PTM_20240825012457_v1.fits"
    path = "/Users/jhughes/Desktop/data/punch build 3 synthetic data/d0/0/PM3/2024/08/25/PUNCH_L0_PM3_20240825001257_v1.fits"
    with fits.open(path) as hdul:
        arr = hdul[1].data[0]

    scaled_arr = (cmap(norm(arr))*255).astype(np.uint8)
    encoded_arr = encode_array(scaled_arr)

    with open("test.j2k", "wb") as f:
        f.write(encoded_arr)
