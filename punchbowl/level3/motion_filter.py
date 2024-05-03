import numpy as np
import scipy.fft
import warnings
from scipy import signal
from skimage.filters import window
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
from sunpy import log

try:
    import cupy

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


# Defining cicular mask
def layer_mask(radius: float, img_shape) -> np.ndarray:
    xs = np.arange(0, img_shape[0])
    ys = np.arange(0, img_shape[1])
    xs, ys = np.meshgrid(xs, ys)
    distance = np.sqrt((xs-img_shape[0]/2)**2 + (ys-img_shape[1]/2)**2)
    return distance < radius

# Creating hourglass filter
def gen_filter(fftcube, cutoff_vel):

    fft_shape = fftcube.shape
    img_shape = (fft_shape[1],fft_shape[2])
    cutoff = cutoff_vel #(*some factor to be estimated next)

    mask1 = np.stack([layer_mask(radius, img_shape) for radius in np.linspace(int(fft_shape[1]/2), cutoff, int(fft_shape[0]/2))], axis=0)
    mask2 = np.stack([layer_mask(radius, img_shape) for radius in np.linspace( cutoff, int(fft_shape[1]/2), int(fft_shape[0]/2))], axis=0)

    filt_mask = np.concatenate((mask1,mask2),axis=0)
    return filt_mask


# Generating motion filtered data
def motion_filter(stacked_data, apod_margin, use_gpu=True):
    padded_data = np.pad(stacked_data, ((apod_margin * 2, apod_margin * 2), (apod_margin, apod_margin),
                                        (apod_margin, apod_margin)), mode='constant', constant_values=0)

    h = window(('tukey', 0.1), padded_data[0].shape)  # Creating a 2D window
    wimage = padded_data * h

    if use_gpu and not HAS_CUPY:
        log.info("cupy not installed or working, falling back to CPU")
    if HAS_CUPY and use_gpu:
        wimage = cupy.ndarray(wimage)

    # Performing 3D FFT
    fwfft = fftshift(fftn(wimage))
    filt_mask = gen_filter(fwfft, 100)
    filt_fft = filt_mask * fwfft  # Adding hourglass mask to Forward FFT

    # Performing inverse FFT
    inv_shift = ifftshift(filt_fft)
    invfft = ifftn(inv_shift)

    return cupy.asnumpy(invfft) if (HAS_CUPY and use_gpu) else invfft