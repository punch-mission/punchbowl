import os
import logging
import threading
import contextlib
import contextvars
import multiprocessing as mp
import logging.handlers
from itertools import repeat

import numpy as np
import scipy.signal
from astropy.coordinates import EarthLocation, get_body
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from ndcube import NDCube
from prefect import get_run_logger
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from punchbowl.data import NormalizedMetadata
from punchbowl.levelq.limits import LimitSet
from punchbowl.prefect import punch_task
from punchbowl.util import DataLoader, load_image_task

# We're suffering two global variables, to facilitate passing information to forker worker processes
_all_files_to_fit = None
_log_queue = None


@punch_task
def pca_filter(input_cubes: list[NDCube], files_to_fit: list[NDCube | DataLoader | str],
               n_components: int=50, med_filt: int=5, outlier_limits: str | LimitSet = None,
               n_strides: int = 8, blend_size: int = 70) -> None:
    """Run PCA-based filtering."""
    logger = get_run_logger()

    if isinstance(outlier_limits, str):
        outlier_limits = LimitSet.from_file(outlier_limits)

    try:
        # We stash the loaded images in a global variable so that, when we fork for parallel processing,
        # those images don't have to be individually copied to each worker process. The try/finally block ensures we
        # don't leak this memory.
        global _all_files_to_fit # noqa: PLW0603
        _all_files_to_fit, bodies_in_quarter, to_subtract, is_masked, is_outlier = load_files(
            input_cubes, files_to_fit, outlier_limits, blend_size)
        # 25 threads per worker would saturate all our cores if they all run at once, but experience shows they don't.
        ctx = mp.get_context("fork")
        with (_log_forwarder(),
              threadpool_limits(min(25, os.cpu_count())),
              ctx.Pool(min(n_strides, os.cpu_count())) as p):
            for subtracted_cube_indices, subtracted_images in p.starmap(
                    pca_filter_one_stride, zip(range(n_strides), repeat(n_strides), repeat(bodies_in_quarter),
                                               repeat(to_subtract), repeat(n_components), repeat(med_filt),
                                               repeat(blend_size), repeat(is_masked), repeat(is_outlier))):
                for index, image in zip(subtracted_cube_indices, subtracted_images, strict=False):
                    input_cubes[index].data[...] = image
    finally:
        _all_files_to_fit = None

    logger.info("PCA filtering finished")


def check_file(meta: NormalizedMetadata, outlier_limits: LimitSet | None) -> bool:
    """Check if a file should be used."""
    return outlier_limits.is_good(meta) if outlier_limits is not None else True


@punch_task
def load_files(input_cubes: list[NDCube], files_to_fit: list[NDCube | str | DataLoader],
               outlier_limits: LimitSet | None = None,
               blend_size: int = 70) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load files."""
    logger = get_run_logger()

    # Join these two sets of things into one list, sorted by observation time, and keep track of which ones need to
    # be subtracted and where they are in the original list of input cubes. We sort by observation time to ensure the
    # staggered dropping of files is spread evenly over time.
    things_to_load = np.array(input_cubes + files_to_fit, dtype=object)
    input_list_indices = np.concatenate((range(len(input_cubes)), [-1] * len(files_to_fit)))
    def sort_key(thing: str | NDCube | DataLoader) -> str:
        if isinstance(thing, str):
            return os.path.basename(thing)
        if isinstance(thing, NDCube):
            return thing.meta["FILENAME"].value
        return os.path.basename(thing.src_repr())
    keys = np.array([sort_key(t) for t in things_to_load])
    sort_by_date = np.argsort(keys)
    things_to_load = things_to_load[sort_by_date]
    input_list_indices = input_list_indices[sort_by_date]

    # We'll pre-allocate an array to load files into. Since we'll reject any bad files, we'll track an insertion
    # index as we add good files, and at the end we'll slice the array to drop any empty spots at the end. When
    # np.empty allocates memory, the OS doesn't *actually* allocate those pages until they're used, so we won't
    # actually use any more RAM than needed.
    all_files_to_fit = np.empty((len(things_to_load), *input_cubes[0].data.shape), dtype=input_cubes[0].data.dtype)
    index_to_insert = 0
    loaded_input_list_indices = []
    n_outliers = 0
    body_finding_inputs = []
    # If a file-to-be-subtracted is an outlier, we want to keep it in the stack so we can still try to subtract it,
    # but we don't want it to factor in to the fitting, so we mark it here.
    is_outlier = []
    # We want to know which pixels are masked in every image. It's possible we'll have files made with two different
    # versions of the mask, so here we detect which pixels are maxed by looking for data == 0 and uncertainty == inf,
    # and we track which pixels satisfy that for every image.
    is_masked = np.ones(input_cubes[0].data.shape, dtype=bool)
    for input_file, subtract in zip(things_to_load, input_list_indices, strict=False):
        if isinstance(input_file, NDCube):
            data, meta = input_file.data, input_file.meta
            body_finding_input = (input_file.meta, input_file.wcs)
            uncertainty_is_inf = np.isinf(input_file.uncertainty.array)
        elif isinstance(input_file, str):
            cube = load_image_task(input_file, include_provenance=False, include_uncertainty=True)
            data, meta = cube.data, cube.meta
            body_finding_input = (cube.meta, cube.wcs)
            uncertainty_is_inf = np.isinf(cube.uncertainty.array)
        elif isinstance(input_file, DataLoader):
            data, meta, wcs, uncertainty_is_inf = input_file.load()
            body_finding_input = (meta, wcs)
        else:
            raise TypeError(f"Invalid type {type(input_file)} for input file")
        is_good = check_file(meta, outlier_limits)
        if is_good or subtract >= 0:
            loaded_input_list_indices.append(subtract)
            all_files_to_fit[index_to_insert] = data
            index_to_insert += 1
            body_finding_inputs.append(body_finding_input)
            is_masked *= uncertainty_is_inf * (data == 0)
            is_outlier.append(not is_good)
            if not is_good:
                n_outliers += 1
        else:
            n_outliers += 1

    loaded_input_list_indices = np.array(loaded_input_list_indices)
    is_outlier = np.array(is_outlier)

    # Crop the unused end of the array
    all_files_to_fit = all_files_to_fit[:index_to_insert]
    logger.info(f"Total of {len(all_files_to_fit)} images to fit, filling {all_files_to_fit.nbytes / 1024 ** 3:.2f} GB")
    logger.info(f"(Drawn from {len(input_cubes)} images to subtract and {len(files_to_fit)} extra images for fitting)")
    logger.info(f"({n_outliers} outliers were rejected from fitting)")

    logger.info("Locating planets")
    # We have a lot of data in memory right now, so forking is expensive as all that memory has to be marked as
    # copy-on-write. Using a forkserver avoids that work.
    ctx = mp.get_context("forkserver")
    with ctx.Pool(min(25, os.cpu_count())) as p:
        bodies_in_quarter = np.array(p.starmap(find_bodies_in_image_quarters,
                                               zip(body_finding_inputs, repeat(blend_size))))

    return all_files_to_fit, bodies_in_quarter, loaded_input_list_indices, is_masked, is_outlier


def pca_filter_one_stride(stride: int, n_strides: int, bodies_in_quarter: np.ndarray, input_list_indices: np.ndarray,
                          n_components: int, med_filt: int, blend_size: int, masked_region: np.ndarray,
                          is_outlier: np.ndarray, logger: logging.Logger | None = None,
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Run PCA-based filtering for one stride position."""
    # This sets up a logger that forwards entries through a queue to the main process, where they can be forwarded to
    # Prefect. This is required because Prefect can't log from forked worker processes.
    if logger is None:
        logger = logging.getLogger()
        logger.addHandler(logging.handlers.QueueHandler(_log_queue))
        logger.setLevel(logging.INFO)

    stride_filter = np.arange(len(_all_files_to_fit)) % n_strides == stride
    # This will mark the images we'll be subtracting from---those are the only ones we'll drop from the fitting
    to_subtract_filter = stride_filter * (input_list_indices >= 0)
    if not np.any(to_subtract_filter):
        logger.info(f"Stride {stride} has no images to subtract")
        return [], []

    images_to_subtract = _all_files_to_fit[to_subtract_filter]
    # This tracks where each image-to-be-subtracted is in the main list of NDCubes
    subtracted_cube_indices = input_list_indices[to_subtract_filter]

    # The quartering approach that protects from planets/the Moon wrecking the PCA components can leave seams. To
    # reduce that, we have a small blend region at those seams. Here we define a mask that's 1 in the core of a
    # quarter and tapers to 0 through the blend region.
    yy, xx = np.indices(images_to_subtract.shape[1:])
    blend_mask = np.clip(((_all_files_to_fit.shape[1] / 2 - 1 + blend_size / 2) - yy) / blend_size, 0, 1)
    blend_mask = blend_mask * blend_mask.T
    # Flip it around to make one for each quarter
    blend_masks = [blend_mask, blend_mask[:, ::-1], blend_mask[::-1], blend_mask[::-1, ::-1]]

    # We need to PCA separately for each quarter of the image
    for i, mask in enumerate(blend_masks):
        # We mark the images that don't have any planets in the quarter we're filtering for (since those can
        # contaminate the PCA components)
        no_bodies_in_quarter = np.all(bodies_in_quarter[:, :, i] == False, axis=1) # noqa: E712
        images_to_fit = _all_files_to_fit[no_bodies_in_quarter * ~to_subtract_filter * ~is_outlier]
        tag = f"stride {stride}, quarter {i+1}"
        logger.info(f"Starting to filter {tag}, fitting {len(images_to_fit)} images")
        filtered_by_quarter = run_pca_filtering(images_to_subtract, images_to_fit, n_components, med_filt, tag,
                                                masked_region, logger)
        if i == 0:
            final_reconstruction = mask * filtered_by_quarter
        else:
            final_reconstruction += mask * filtered_by_quarter

    return subtracted_cube_indices, final_reconstruction


def run_pca_filtering(images_to_subtract: np.ndarray, images_to_fit: np.ndarray, n_components: int,
                      med_filt: int, tag: str, masked_region: np.ndarray, logger: logging.Logger) -> np.ndarray:
    """Run PCA filtering."""
    pca = PCA(n_components=n_components)
    # The image array has to be re-shaped into (n_images, n_pixels). (i.e., PCA wants 1D vectors, not 2D images)
    pca.fit(images_to_fit[:, masked_region])
    logger.info(f"Fitting finished for {tag}")

    transformed = pca.transform(images_to_subtract[:, masked_region])

    if med_filt:
        for i in range(len(pca.components_)):
            comp = np.zeros(images_to_fit.shape[1:], dtype=pca.components_[i].dtype)
            comp[masked_region] = pca.components_[i]
            comp = scipy.signal.medfilt2d(comp, med_filt)
            pca.components_[i] = comp[masked_region]
        logger.info(f"Median smoothing finished for {tag}")

    reconstructed = np.zeros_like(images_to_subtract)
    reconstructed[:, masked_region] = pca.inverse_transform(transformed)
    return images_to_subtract - reconstructed


def find_bodies_in_image_quarters(frame: str | NDCube | tuple[NormalizedMetadata, WCS], extra_padding: int = 0) -> list:
    """Find celestial bodies in image."""
    if isinstance(frame, str):
        header = fits.getheader(frame, 1)
        wcs = WCS(header)
        location = header["GEOD_LON"], header["GEOD_LAT"], header["GEOD_ALT"]
        image_shape = header["NAXIS2"], header["NAXIS1"]
    elif isinstance(frame, NDCube):
        location = frame.meta["GEOD_LON"].value, frame.meta["GEOD_LAT"].value, frame.meta["GEOD_ALT"].value
        wcs = frame.wcs
        image_shape = frame.data.shape
    elif isinstance(frame, tuple):
        meta, wcs = frame
        location = meta["GEOD_LON"].value, meta["GEOD_LAT"].value, meta["GEOD_ALT"].value
        image_shape = wcs.array_shape
    else:
        msg = "Type of 'frame' not recognized"
        raise TypeError(msg)

    results = []
    for body in ["Mercury", "Venus", "Moon", "Mars", "Jupiter", "Saturn"]:
        body_loc = get_body(body, time=Time(wcs.wcs.dateobs), location=EarthLocation.from_geodetic(*location))
        x, y = wcs.world_to_pixel(body_loc)
        # Extra margin for the big moon
        w = 100 if body == "Moon" else 10
        w += extra_padding
        in_left = 0 - w <= x <= image_shape[1] / 2 + w
        in_right = image_shape[1] / 2 - w <= x <= image_shape[1] + w
        in_bottom = 0 - w <= y <= image_shape[0] / 2 + w
        in_top = image_shape[0] / 2 - w <= y <= image_shape[0] + w
        body_in_quarter = [
            in_left and in_bottom,
            in_right and in_bottom,
            in_left and in_top,
            in_right and in_top]
        results.append(body_in_quarter)
    return results


@contextlib.contextmanager
def _log_forwarder() -> None:
    # This logging situation is kind of a mess. We really want to parallelize with multiprocessing so we can fork and
    # not copy all the images to each worker. But Prefect logging from those forked processes doesn't work (nor does
    # logging print statements), so in the workers we need to set up a logger than puts log entries in a
    # shared-memory Queue, and then on the main process side we need to spawn a thread that monitors the queue and
    # forwards log entries to Prefect.

    # The log queue has to be a global variable so the worker processes can retrieve it
    global _log_queue # noqa: PLW0603

    def _logger_consumer_thread(logging_q: mp.Queue) -> None:
        # This gets the right Prefect metadata attached to our forwarded log entries
        current_thread_context = dict(contextvars.copy_context().items())

        logger = get_run_logger()

        while True:
            record = logging_q.get()
            if record is None:
                break
            logger.log(level=record.levelno, msg=record.msg, extra=current_thread_context)

    try:
        _log_queue = mp.Queue()
        # This is part of getting the right Prefect metadata attached to our forwarded log entries
        current_thread_context = contextvars.copy_context()

        logging_q_consumer_thread = threading.Thread(
            target=current_thread_context.run,
            args=(lambda: _logger_consumer_thread(_log_queue),),
            # Daemonizing the thread allows the Python process to quit when all non-daemonized threads terminate
            daemon=True,
        )
        logging_q_consumer_thread.start()
        yield
    finally:
        try: # noqa: SIM105
            # Signal to the thread it can quit
            _log_queue.put(None)
        except: # noqa: S110, E722
            # Don't let problems here stop us from clearing the global variable
            pass
        _log_queue = None
