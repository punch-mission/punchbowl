import numpy as np
from ndcube import NDCube
from prefect import get_run_logger, task


def dn_to_photons(data_array: np.ndarray, gain: float = 4.3) -> np.ndarray:
    """Convert an input array from DN to photon count."""
    return data_array * gain


def compute_noise(
        data: np.ndarray,
        bias_level: float = 100,
        dark_level: float = 55.81,
        gain: float = 4.9,
        read_noise_level: float = 17,
        bitrate_signal: int = 16) -> np.ndarray:
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
    gain
        ccd gain
    read_noise_level
        ccd read noise level
    bitrate_signal
        desired ccd data bit level

    Returns
    -------
    np.ndarray
        computed noise array corresponding to input data and ccd/noise parameters

    """
    data = data.copy().astype("long")

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
    return noise_photon + noise_dark + noise_read


def compute_uncertainty(data_array: np.ndarray,
                        bias_level: float = 100,
                        dark_level: float = 55.81,
                        gain: float = 4.9,
                        read_noise_level: float = 17,
                        bitrate_signal: int = 16,
                        ) -> np.ndarray:
    """With an input data array compute a corresponding uncertainty array."""
    # Convert this photon count to a shot noise
    return compute_noise(data_array,
                                bias_level=bias_level,
                                dark_level=dark_level,
                                gain=gain,
                                read_noise_level=read_noise_level,
                                bitrate_signal=bitrate_signal)


@task
def update_initial_uncertainty_task(data_object: NDCube,
                                    bias_level: float = 100,
        dark_level: float = 55.81,
        gain: float = 4.9,
        read_noise_level: float = 17,
        bitrate_signal: int = 16) -> NDCube:
    """Prefect task to compute initial uncertainty."""
    logger = get_run_logger()
    logger.info("initial uncertainty computation started")

    uncertainty_array = compute_uncertainty(data_object.data,
                                            bias_level=bias_level,
                                            dark_level=dark_level,
                                            gain=gain,
                                            read_noise_level=read_noise_level,
                                            bitrate_signal=bitrate_signal,
                                            )
    data_object.uncertainty.array = uncertainty_array

    data_object.meta.history.add_now("LEVEL1-initial_uncertainty", "Initial uncertainty computed with:")
    data_object.meta.history.add_now("LEVEL1-initial_uncertainty", f"bias_level={bias_level}")
    data_object.meta.history.add_now("LEVEL1-initial_uncertainty", f"dark_level={dark_level}")
    data_object.meta.history.add_now("LEVEL1-initial_uncertainty", f"gain={gain}")
    data_object.meta.history.add_now("LEVEL1-initial_uncertainty", f"read_noise_level={read_noise_level}")
    data_object.meta.history.add_now("LEVEL1-initial_uncertainty", f"bitrate_signal={bitrate_signal}")

    logger.info("initial uncertainty computed")
    return data_object
