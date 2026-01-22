from concurrent.futures import ThreadPoolExecutor

import numpy as np
from ndcube import NDCube
from prefect import get_run_logger
from scipy.ndimage import gaussian_filter
from threadpoolctl import threadpool_limits

from punchbowl.data import load_ndcube_from_fits
from punchbowl.level1.deficient_pixel import mean_correct
from punchbowl.level1.sqrt import decode_sqrt_data
from punchbowl.prefect import punch_task


def despike_polseq(
        reference: NDCube,
        neighbors: list[NDCube],
        filter_width: float=25.0,
        hpf_zscore_thresh: float=10.0,
)->tuple[NDCube, np.ndarray]:
    """
    Remove cosmic ray spikes from a single polarization sequence of images.

    This code takes as input multiple (N) images from the same roll sequence. It
    constructs a high-pass-filtered version of the input images. At each pixel, it
    computes the median and standard deviation of the (N-1) dimmest pixels, and
    then the z-score of each pixel. If the z-score exceeds a threshold, a cosmic
    ray is assumed and the pixel is filled in with the mean of its neighbors.

    Parameters
    ----------
    reference : NDCube
        an NDCube to correct for cosmic rays
    neighbors : List[NDCube]
        a list of NDCube objects representing a polarization image sequence, should not include the reference image
    filter_width: float
        width of the gaussian filter used in created the high-pass-filtered image
    hpf_zscore_thresh: float
        number of standard deviations above the sequence median[sic] that causes a pixel to be marked as a cosmic ray.

    Returns
    -------
    (NDCube, np.ndarray)
        a  NDCube with spikes replaced by the average of their neighbors,
         and a list of spike locations for all neighbors

    """
    sequence = np.stack([cube.data for cube in [*neighbors, reference]], axis=0)
    seq_len = sequence.shape[0]

    # create the high-pass-filtered images
    def blur_one_image(image: np.ndarray)->np.ndarray:
        return gaussian_filter(image, [filter_width,filter_width], mode="nearest")

    with ThreadPoolExecutor(len(sequence)) as p:
        lpf_decoded = np.stack(list(p.map(blur_one_image, sequence)))

    hpf = sequence.astype(float) - lpf_decoded.astype(float)
    hpf_sorted = np.sort(hpf, axis=0)

    #For a polarization sequence of length N, this finds the median
    #of the N-1 lowest values of each pixel
    match seq_len:
        case 7:
            hpf_median_s = np.mean(hpf_sorted[2:4],axis=0)
        case 6:
            hpf_median_s = hpf_sorted[2]
        case 5:
            hpf_median_s = np.mean(hpf_sorted[1:3],axis=0)
        case 4:
            hpf_median_s = hpf_sorted[1]
        case 3:
            hpf_median_s = np.mean(hpf_sorted[0:2],axis=0)
        case _:
            raise RuntimeError(f"A sequence length of {seq_len} is not supported.")

    hpf_stdev = np.std(hpf_sorted[:-1], axis=0, ddof=0)
    hpf_zscore = (hpf - hpf_median_s)/hpf_stdev

    cosmic_sequence = np.zeros_like(sequence, dtype=bool)
    cosmic_sequence[hpf_zscore>=hpf_zscore_thresh] = 1

    reference.data[(hpf_zscore>=hpf_zscore_thresh)[-1]] = np.nan

    reference.data = mean_correct(data_array=reference.data, mask_array=~np.isnan(reference.data))

    return reference, cosmic_sequence


@punch_task
def despike_polseq_task(data_object: NDCube,
                        neighbors: list[NDCube],
                        filter_width: float=25.0,
                        hpf_zscore_thresh: float=10.0,
                        max_workers: int | None = None)-> NDCube:
    """
    Despike a polarization sequence of images using a simple statistical test.

    Parameters
    ----------
    data_object : NDCube
        Image to be despiked.
    neighbors : list[NDCube]
        Sequence of neighbor images from the same spacecraft and roll sequence to use in despiking.
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
    logger = get_run_logger()

    if neighbors is not None:
        logger.info(f"Neighbors = {neighbors}")
        neighbors = [decode_sqrt_data(load_ndcube_from_fits(n)) if isinstance(n, str) else n for n in neighbors]

        with threadpool_limits(max_workers):
            data_object, spikes = despike_polseq(
                                            data_object,
                                            neighbors,
                                            filter_width=filter_width,
                                            hpf_zscore_thresh=hpf_zscore_thresh)

        data_object.uncertainty.array[spikes[-1]] = np.inf
        data_object.meta.history.add_now("LEVEL1-despike", "image despiked")
        data_object.meta.history.add_now("LEVEL1-despike", f"filter_width={filter_width}")
        data_object.meta.history.add_now("LEVEL1-despike", f"zscore_thresh={hpf_zscore_thresh}")
        data_object.meta.history.add_now("LEVEL1-despike", f"neighbor_count={len(neighbors)}")
    else:
        data_object.meta.history.add_now("LEVEL1-despike", "Empty neighbors so no correction applied")
        logger.info("No polarization neighbors so despiking is skipped")
    return data_object
