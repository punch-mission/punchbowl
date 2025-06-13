import numpy as np
from astropy.nddata import StdDevUncertainty
from ndcube import NDCube

from punchbowl.data.units import split_ccd_array
from punchbowl.prefect import punch_task


def dn_to_photons(data_array: np.ndarray, gain_bottom: float = 4.9, gain_top: float = 4.9) -> np.ndarray:
    """Convert an input array from DN to photon count."""
    gain = split_ccd_array(data_array.shape, gain_bottom, gain_top)
    return data_array * gain


def compute_noise(
        data: np.ndarray,
        bias_level: float = 100,
        dark_level: float = 55.81,
        gain_bottom: float = 4.9,
        gain_top: float = 4.9,
        read_noise_level: float = 17,
        bitrate_signal: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate noise based on an input data array, with specified noise parameters.

    Parameters
    ----------
    data
        input data array (n x n)
    bias_level
        ccd bias level
    dark_level
        ccd dark level
    gain_bottom
        ccd gain (bottom side of CCD)
    gain_top
        ccd gain (top side of CCD)
    read_noise_level
        ccd read noise level
    bitrate_signal
        desired ccd data bit level

    Returns
    -------
    data : np.ndarray
        clipped data with the bias level added
    noise : np.ndarray
        computed noise array corresponding to input data and ccd/noise parameters

    """
    data = data.copy().astype("long")

    gain = split_ccd_array(data.shape, gain_bottom, gain_top)

    # Photon / shot noise generation
    data_photon = data * gain  # DN to photoelectrons
    sigma_photon = np.sqrt(data_photon)  # Converting sigma of this
    sigma = sigma_photon / gain  # Converting back to DN
    noise_photon = np.random.normal(scale=sigma)

    # Add bias level and clip pixels to avoid overflow
    data = np.clip(data + bias_level, 0, 2 ** bitrate_signal - 1)

    # Dark noise generation
    noise_level = dark_level * gain
    noise_dark = np.random.poisson(lam=noise_level, size=data.shape) / gain

    # Read noise generation
    noise_read = np.random.normal(scale=read_noise_level, size=data.shape)
    noise_read = noise_read / gain  # Convert back to DN

    # And then add noise terms directly
    return data, noise_photon + noise_dark + noise_read


def compute_uncertainty(data_array: np.ndarray,
                        dark_level: float = 55.81,
                        gain_bottom: float = 4.9,
                        gain_top: float = 4.9,
                        read_noise_level: float = 17,
                        ) -> np.ndarray:
    """With an input data array compute a corresponding uncertainty array."""
    # Convert this photon count to a shot noise
    data = data_array.copy()

    gain = split_ccd_array(data.shape, gain_bottom, gain_top)

    # Photon / shot noise generation
    data_photon = data * gain  # DN to photoelectrons
    sigma_photon = np.sqrt(data_photon)  # Converting sigma of this
    noise_photon = sigma_photon / gain  # Converting back to DN

    # Dark noise generation
    noise_dark = dark_level

    # Read noise generation
    noise_read = read_noise_level / gain  # Convert back to DN

    # And then add noise terms directly
    return np.sqrt(noise_photon**2 + noise_dark**2 + noise_read**2)


def flag_saturated_pixels(data_object: NDCube,
                          saturated_pixels: np.ndarray) -> NDCube:
    """Flag saturated pixels in the uncertainty layer."""
    if saturated_pixels is not None:
        data_object.uncertainty.array[saturated_pixels] = np.inf
    return data_object


@punch_task
def update_initial_uncertainty_task(data_object: NDCube,
                                    dark_level: float = 55.81,
                                    gain_bottom: float = 4.9,
                                    gain_top: float = 4.9,
                                    read_noise_level: float = 17,
                                    bitrate_signal: int = 16,
                                    saturated_pixels: np.ndarray | None = None) -> NDCube:
    """Prefect task to compute initial uncertainty."""
    uncertainty_array = compute_uncertainty(data_object.data,
                                            dark_level=dark_level,
                                            gain_bottom=gain_bottom,
                                            gain_top=gain_top,
                                            read_noise_level=read_noise_level,
                                            )
    data_object.uncertainty = StdDevUncertainty(uncertainty_array)

    flag_saturated_pixels(data_object, saturated_pixels)

    data_object.meta.history.add_now("LEVEL1-initial_uncertainty", "Initial uncertainty computed with:")
    data_object.meta.history.add_now("LEVEL1-initial_uncertainty", f"dark_level={dark_level}")
    data_object.meta.history.add_now("LEVEL1-initial_uncertainty", f"gain_bottom={gain_bottom}")
    data_object.meta.history.add_now("LEVEL1-initial_uncertainty", f"gain_top={gain_top}")
    data_object.meta.history.add_now("LEVEL1-initial_uncertainty", f"read_noise_level={read_noise_level}")
    data_object.meta.history.add_now("LEVEL1-initial_uncertainty", f"bitrate_signal={bitrate_signal}")
    return data_object
