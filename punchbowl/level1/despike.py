import numpy as np
import numpy.ma as ma  # DAL
from astroscrappy import detect_cosmics  # DAL remove
from ndcube import NDCube
from scipy.ndimage import gaussian_filter  # DAL
from scipy.signal import convolve2d, medfilt2d
from threadpoolctl import threadpool_limits

from punchbowl.level1.deficient_pixel import cell_neighbors  # needed for spikejones
from punchbowl.level1.deficient_pixel import mean_correct  # DAL
from punchbowl.prefect import punch_task


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
) -> tuple[np.ndarray, np.ndarray]:
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
        an image with spikes replaced by the average of their neighbors and the spike locations of all spikes

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
        unsharped_image = convolve2d(unsharped_image, np.ones((dilation_size, dilation_size)), mode="same") != 0

    # detect the spikes and fill them with their neighbors
    spikes = np.where(unsharped_image != 0)
    output = np.copy(image)
    image[spikes] = np.nan
    for x, y in zip(*spikes, strict=False):
        neighbors = cell_neighbors(image, x, y, kernel_size - 1)
        threshold = np.nanmedian(neighbors) + 3 * np.nanstd(neighbors)
        output[x, y] = np.nanmean(neighbors[neighbors < threshold])

    return output, spikes

@punch_task
def despike_polseq(
    sequence: list[NDCube], sat_ratio: float=0.99, filter_width: float=25.0, hpf_zscore_thresh: float=20.0,
    )->tuple[list[NDCube],list[np.ndarray]]:
    """
    Remove cosmic ray spikes from a single polarization sequence of images.

    This code takes as input multiple (N) images from the same roll sequence. It
    constructs a high-pass-filtered version of the input images. At each pixel, it
    computes the median and standard deviation of the (N-1) dimmest pixels, and
    then the z-score of each pixel. If the z-score exceeds a threshold, a cosmic
    ray is assumed and the pixel is filled in with the mean of its neighbors.

    Parameters
    ----------
    sequence : List[NDCube]
        a list of NDCube objects representing a polarization image sequence
    sat_ratio: float
        pixels greater than this fraction of the saturation value are set to NaN
    filter_width: float
        width of the gaussian filter used in created the high-pass-filtered image
    hpf_zscore_thresh: float
        number of standard deviations above the sequence median[sic] that causes a pixel to be marked as a cosmic ray.

    Returns
    -------
    (List[NDCube], List[np.ndarray])
        a list of NDCubes with spikes replaced by the average of their neighbors, and a list of spike locations

    """
    seq_len = sequence.shape[0]
    dsatval = 65535 #DAL this assumes we move cosmic removal before the DN->MSB conversion
    sequence[sequence>=sat_ratio*dsatval]=np.nan

    #create the high-pass-filtered images
    lpf_decoded = gaussian_filter(sequence,[1,filter_width,filter_width],mode="nearest")
    lpf_decoded = ma.filled(lpf_decoded,np.nan)
    hpf = sequence-lpf_decoded #DAL need to make sure hpf gets to be a signed datatype

    hpf_sorted = np.sort(hpf,axis=0)

    match(seq_len):
        case(7):
            hpf_median_s = np.mean(hpf_sorted[2:3],axis=0)
        case(6):
            hpf_median_s = hpf_sorted[2]
        case(5):
            hpf_median_s = np.mean(hpf_sorted[1:2],axis=0)
        case(4):
            hpf_median_s = hpf_sorted[1]
        case(3):
            hpf_median_s = np.mean(hpf_sorted[0:1],axis=0)

    hpf_stdev = np.std(hpf_sorted[0:-1],axis=0,ddof=0)
    hpf_zscore = (hpf - hpf_median_s)/hpf_stdev

    sequence_replaced = np.array(sequence)
    cosmic_sequence = np.zeroes(sequence_replaced)
    cosmic_sequence[hpf_zscore>=hpf_zscore_thresh]=1
    sequence_replaced[hpf_zscore>=hpf_zscore_thresh]=np.nan

    sequence_replaced = [mean_correct(data_array=sequence_replaced[_],mask_array=
                                      np.logical_not(np.isnan(sequence_replaced[_]))) for _ in range(seq_len)]

    return sequence_replaced, cosmic_sequence


@punch_task
def despike_polseq_task(data_object: NDCube,
                            sat_ratio: float=0.99,
                            filter_width: float=25.0,
                            hpf_zscore_thresh: float=20.0,
                            max_workers: int | None = None)-> NDCube:
    """
    Despike a polarization sequence of images using a simple statistical test.

    Parameters
    ----------
    data_object : NDCube
        Sequence of images to be despiked. Must be from the same spacecraft and roll sequence.
    sat_ratio: float, optional
        Pixels greater than sat_ratio times the saturation value are set to NaN.
    filter_width : float, optional
        width of the gaussian filter used to construct the high-pass-filtered image, in pixels.
    hpf_zscore_thresh: float, optional
        number of standard deviations above the sequence median[sic] that causes a pixel to be marked as a cosmic ray.
    max_workers : int, optional
        Max number of threads to use

    Returns
    -------
    NDCube
        Despiked cube.

    """
    with threadpool_limits(max_workers):
        data_object.data[...], spikes = despike_polseq(
                                        data_object.data[...],
                                        sat_ratio=sat_ratio,
                                        filter_width=filter_width,
                                        hpf_zscore_thresh=hpf_zscore_thresh)

    data_object.uncertainty.array[spikes] = np.inf
    data_object.meta.history.add_now("LEVEL1-despike", "image despiked")
    data_object.meta.history.add_now("LEVEL1-despike", f"saturation_ratio={sat_ratio}")
    data_object.meta.history.add_now("LEVEL1-despike", f"filter_width={filter_width}")
    data_object.meta.history.add_now("LEVEL1-despike", f"zscore_thresh={hpf_zscore_thresh}")

    return data_object

@punch_task
def despike_task(data_object: NDCube,
                        sigclip: float=50,
                        sigfrac: float=0.25,
                        objlim: float=160.0,
                        niter:int=10,
                        gain:float=4.9,
                        readnoise:float=17,
                        cleantype:str="meanmask",
                        max_workers: int | None = None)-> NDCube:
    """
    Despike an image using astroscrappy.detect_cosmics.

    Parameters
    ----------
    data_object : NDCube
        Input image to be despiked.
    sigclip : float, optional
        Laplacian-to-noise limit for cosmic ray detection.
    sigfrac : float, optional
        Fractional detection limit for neighboring pixels.
    objlim : float, optional
        Contrast limit between Laplacian image and the fine structure image.
    niter : int, optional
        Number of iterations.
    gain : float, optional
        Gain of the image (electrons/ADU).
    readnoise : float, optional
        Read noise of the image (electrons).
    cleantype : str, optional
        Type of cleaning algorithm: 'meanmask', 'medmask', or 'idw'.
    max_workers : int, optional
        Max number of threads to use

    Returns
    -------
    NDCube
        Despiked cube.

    """
    with threadpool_limits(max_workers):
        spikes, data_object.data[...] = detect_cosmics(
                                        data_object.data[...],
                                        sigclip=sigclip,
                                        sigfrac=sigfrac,
                                        objlim=objlim,
                                        niter=niter,
                                        gain=gain,
                                        readnoise=readnoise,
                                        cleantype=cleantype)

    data_object.uncertainty.array[spikes] = np.inf
    data_object.meta.history.add_now("LEVEL1-despike", "image despiked")
    data_object.meta.history.add_now("LEVEL1-despike", f"method={cleantype}")
    data_object.meta.history.add_now("LEVEL1-despike", f"sigclip={sigclip}")
    data_object.meta.history.add_now("LEVEL1-despike", f"unsharp_size={sigfrac}")
    data_object.meta.history.add_now("LEVEL1-despike", f"alpha={objlim}")
    data_object.meta.history.add_now("LEVEL1-despike", f"iterations={niter}")

    return data_object



@punch_task
def despikejones_task(data_object: NDCube,
                 unsharp_size: int = 3,
                 method: str = "convolve",
                 alpha: float = 1,
                 dilation: int = 0) -> NDCube:
    """
    Prefect task to perform despiking.

    Parameters
    ----------
    data_object : NDCube
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
    NDCube
        a modified version of the input with spikes removed

    """
    data_object.data[...], spikes = spikejones(
        data_object.data[...], unsharp_size=unsharp_size, method=method, alpha=alpha, dilation=dilation,
    )
    data_object.uncertainty.array[spikes] = np.inf
    data_object.meta.history.add_now("LEVEL1-despike", "image despiked")
    data_object.meta.history.add_now("LEVEL1-despike", f"method={method}")
    data_object.meta.history.add_now("LEVEL1-despike", f"unsharp_size={unsharp_size}")
    data_object.meta.history.add_now("LEVEL1-despike", f"alpha={alpha}")
    data_object.meta.history.add_now("LEVEL1-despike", f"dilation={dilation}")

    return data_object
