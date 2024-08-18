import numpy as np
from ndcube import NDCube
from prefect import get_run_logger, task
from scipy.signal import convolve2d, medfilt2d

from punchbowl.level1.deficient_pixel import cell_neighbors


def radial_array(shape: tuple[int], center: tuple[int] | None = None) -> np.ndarray:
    """Create radial array."""
    if len(shape) != 2:
        msg = f"Shape must be 2D, received {shape} with {len(shape)} dimensions"
        raise ValueError(msg)

    x = np.arange(shape[0])
    y = np.arange(shape[1])
    X, Y = np.meshgrid(x, y)  # noqa: N806

    center = [s // 2 for s in shape] if center is None else center
    return np.floor(np.sqrt(np.square(X - center[0]) + np.square(Y - center[1])))



def spikejones(
    image: np.ndarray, unsharp_size: int = 3, method: str = "convolve", alpha: float = 1, dilation: int = 0,
) -> (np.ndarray, np.ndarray):
    """
    Remove cosmic ray spikes from an image using spikejones algorithm.

    This code is based on https://github.com/drzowie/solarpdl-tools/blob/master/image/spikejones.pdl

    Parameters
    ----------
    image : np.ndarray
        an array representing an image
    unsharp_size : int
        half window size in pixels for unsharp mask
    method : str (either "convolve" or "median")
        method for applying the unsharp mask
    alpha : float
        threshold for deciding a pixel is a cosmic ray spike, i.e. difference between unsharp and smoothed image
    dilation : int
        how many times to dilate pixels identified as spikes, allows for identifying a larger spike region

    Returns
    -------
    (np.ndarray, np.ndarray)
        an image with spikes replaced by the average of their neighbors and the locations of all spikes

    """
    image = image.copy()  # copy to avoid mutating the existing data
    # compute the sizes and smoothing kernel to be used
    kernel_size = unsharp_size * 2 + 1
    smoothing_size = kernel_size * 2 + 1
    smoothing_kernel = radial_array((smoothing_size, smoothing_size)) <= (smoothing_size / 2)
    smoothing_kernel = smoothing_kernel / np.sum(smoothing_kernel)

    # depending on the method, perform unsharping
    if method == "median":
        normalized_image = image / medfilt2d(image, kernel_size=smoothing_size)
        unsharped_image = normalized_image - medfilt2d(normalized_image, kernel_size=kernel_size) > alpha
    elif method == "convolve":
        normalized_image = image / convolve2d(image, smoothing_kernel, mode="same")
        unsharp_kernel = -np.ones((kernel_size, kernel_size)) / kernel_size / kernel_size
        unsharp_kernel[kernel_size // 2, kernel_size // 2] += 0.75
        unsharp_kernel[kernel_size // 2 - 1: kernel_size // 2 + 1, kernel_size // 2 - 1: kernel_size // 2 + 1] += (
            0.25 / 9
        )
        unsharped_image = convolve2d(normalized_image, unsharp_kernel, mode="same") > alpha
    else:
        msg = f"Unsupported method. Method must be 'median' or 'convolve' but received {method}"
        raise NotImplementedError(msg)

    # optional dilation
    if dilation != 0:
        dilation_size = 2 * dilation + 1
        unsharped_image = convolve2d(unsharped_image, np.ones(dilation_size, dilation_size), mode="same") != 0

    # detect the spikes and fill them with their neighbors
    spikes = np.where(unsharped_image != 0)
    output = np.copy(image)
    image[spikes] = np.nan
    for x, y in zip(*spikes, strict=False):
        output[x, y] = np.nanmean(cell_neighbors(image, x, y, kernel_size - 1))

    return output, spikes


@task
def despike_task(data_object: NDCube,
                 unsharp_size: int = 3,
                 method: str = "convolve",
                 alpha: float = 1,
                 dilation: int = 0) -> NDCube:
    """
    Prefect task to perform despiking.

    Parameters
    ----------
    data_object : PUNCHData
        data to operate on
    unsharp_size : int
        half window size in pixels for unsharp mask
    method : str (either "convolve" or "median")
        method for applying the unsharp mask
    alpha : float
        threshold for deciding a pixel is a cosmic ray spike, i.e. difference between unsharp and smoothed image
    dilation : int
        how many times to dilate pixels identified as spikes, allows for identifying a larger spike region

    Returns
    -------
    PUNCHData
        a modified version of the input with spikes removed

    """
    logger = get_run_logger()
    logger.info("despike started")
    data_object.data[...], spikes = spikejones(
        data_object.data[...], unsharp_size=unsharp_size, method=method, alpha=alpha, dilation=dilation,
    )
    data_object.uncertainty.array[spikes] = np.inf
    logger.info("despike finished")
    data_object.meta.history.add_now("LEVEL1-despike", "image despiked")
    data_object.meta.history.add_now("LEVEL1-despike", f"method={method}")
    data_object.meta.history.add_now("LEVEL1-despike", f"unsharp_size={unsharp_size}")
    data_object.meta.history.add_now("LEVEL1-despike", f"alpha={alpha}")
    data_object.meta.history.add_now("LEVEL1-despike", f"dilation={dilation}")

    return data_object
