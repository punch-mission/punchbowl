import numpy as np
from scipy.fftpack import fftn, fftshift, ifftn, ifftshift
from skimage.filters import window
from sunpy import log

try:
    import cupy
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def layer_mask(radius: float, img_shape: (int, int)) -> np.ndarray:
    """
    Define a circular mask.

    Parameters
    ----------
    radius : float
        The radius of the circular mask.
    img_shape : tuple
        The shape of the input image.

    Returns
    -------
    np.ndarray
        Circular mask.

    """
    xs = np.arange(0, img_shape[0])
    ys = np.arange(0, img_shape[1])
    xs, ys = np.meshgrid(xs, ys)
    distance = np.sqrt((xs - img_shape[0] / 2) ** 2 + (ys - img_shape[1] / 2) ** 2)
    return distance < radius


def generate_hourglass_filter(fft_cube: np.ndarray, cutoff_velocity: float) -> np.ndarray:
    """
    Create an hourglass filter mask.

    Parameters
    ----------
    fft_cube : np.ndarray
        The 3D Fourier space cube.
    cutoff_velocity : float
        The cutoff velocity.

    Returns
    -------
    np.ndarray
        Hourglass filter mask.

    """
    fft_shape = fft_cube.shape
    img_shape = (fft_shape[1], fft_shape[2])

    mask1 = np.stack(
        [layer_mask(radius, img_shape)
         for radius in np.linspace(int(fft_shape[1] / 2), cutoff_velocity, int(fft_shape[0] / 2))],
        axis=0)
    mask2 = np.stack(
        [layer_mask(radius, img_shape)
         for radius in np.linspace(cutoff_velocity, int(fft_shape[1] / 2), int(fft_shape[0] / 2))],
        axis=0)

    return np.concatenate((mask1, mask2), axis=0)


def apply_motion_filter(stacked_data:np.ndarray,
                  apod_margin: int,
                  use_gpu: bool = True) -> np.ndarray:
    """
    Perform a Fourier motion filter on input datacube.

    Parameters
    ----------
    stacked_data : np.ndarray
        Starfield removed stacked datacube.
    apod_margin : int
        Apodization margin.
    use_gpu : bool, optional
        Whether to use GPU for processing (default is True).

    Returns
    -------
    np.ndarray
        Fourier motion filtered datacube.

    """
    padded_data = np.pad(stacked_data, ((apod_margin * 2, apod_margin * 2), (apod_margin, apod_margin),
                                        (apod_margin, apod_margin)), mode="constant", constant_values=0)

    h = window(("tukey", 0.1), padded_data[0].shape)  # Creating a 2D window
    wimage = padded_data * h

    if use_gpu and not HAS_CUPY:
        log.info("cupy not installed or working, falling back to CPU")
    if HAS_CUPY and use_gpu:
        wimage = cupy.ndarray(wimage)

    # Performing 3D FFT
    fwfft = fftshift(fftn(wimage))
    filt_mask = generate_hourglass_filter(fwfft, 100)
    filt_fft = filt_mask * fwfft  # Adding hourglass mask to Forward FFT

    # Performing inverse FFT
    inv_shift = ifftshift(filt_fft)
    invfft = ifftn(inv_shift)

    return cupy.asnumpy(invfft) if (HAS_CUPY and use_gpu) else invfft
