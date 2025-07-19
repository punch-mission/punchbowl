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

_all_files_to_fit = None
_log_queue = None


@punch_task
def pca_filter(input_cubes: list[NDCube], files_to_fit: list[NDCube | DataLoader | str],
               n_components: int=50, med_filt: int=5, outlier_limits: str | LimitSet = None,
               n_strides: int = 8) -> None:
    """Run PCA-based filtering."""
    logger = get_run_logger()

    if isinstance(outlier_limits, str):
        outlier_limits = LimitSet.from_file(outlier_limits)

    try:
        # We stash the loaded images in a global variable so that, when we fork for parallel processing,
        # those images don't have to be individually copied to each worker process.
        global _all_files_to_fit # noqa: PLW0603
        _all_files_to_fit, bodies_in_quarter, to_subtract = load_files(input_cubes, files_to_fit, outlier_limits)
        # 25 threads per worker would saturate all our cores if they all run at once, but experience shows they don't.
        with _log_forwarder(), threadpool_limits(25), mp.Pool(n_strides) as p:
            for subtracted_cube_indices, subtracted_images in p.starmap(
                    pca_filter_one_stride, zip(range(n_strides), repeat(n_strides), repeat(bodies_in_quarter),
                    repeat(to_subtract), repeat(n_components), repeat(med_filt))):
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
               outlier_limits: LimitSet | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load files."""
    logger = get_run_logger()

    # Join these two sets of things into one list, sorted by observation time, and keep track of which ones need to
    # be subtracted and where they are in the original list of input cubes. We sort by observation time to ensure the
    # staggered dropping of files is spread evenly over time.
    things_to_load = np.array(input_cubes + files_to_fit, dtype=object)
    to_subtract = np.concatenate((range(len(input_cubes)), [-1] * len(files_to_fit)))
    def sort_key(thing: str | NDCube | DataLoader) -> str:
        if isinstance(thing, str):
            return os.path.basename(thing)
        if isinstance(thing, NDCube):
            return thing.meta["FILENAME"].value
        return os.path.basename(thing.src_repr())
    keys = np.array([sort_key(t) for t in things_to_load])
    sort_by_date = np.argsort(keys)
    things_to_load = things_to_load[sort_by_date]
    to_subtract = to_subtract[sort_by_date]

    # We'll pre-allocate an array to load files into. Since we'll reject any bad files, we'll track an insertion
    # index as we add good files, and at the end we'll slice the array to drop any empty spots at the end. When
    # np.empty allocates memory, the OS doesn't *actually* allocate those pages until they're used, so we won't
    # actually use any more RAM than needed.
    all_files_to_fit = np.empty((len(things_to_load), *input_cubes[0].data.shape), dtype=input_cubes[0].data.dtype)
    index_to_insert = 0
    bodies_in_quarter = []
    loaded_to_subtract = []
    n_outliers = 0
    for input_file, subtract in zip(things_to_load, to_subtract, strict=False):
        if isinstance(input_file, NDCube):
            data, meta = input_file.data, input_file.meta
            bodies = find_bodies_in_image(input_file)
        elif isinstance(input_file, str):
            cube = load_image_task(input_file, include_provenance=False, include_uncertainty=False)
            data, meta = cube.data, cube.meta
            bodies = find_bodies_in_image(cube)
        elif isinstance(input_file, DataLoader):
            data, meta, bodies = input_file.load()
        else:
            raise TypeError(f"Invalid type {type(input_file)} for input file")
        if check_file(meta, outlier_limits):
            loaded_to_subtract.append(subtract)
            all_files_to_fit[index_to_insert] = data
            index_to_insert += 1
            bodies_in_quarter.append(bodies)
        else:
            n_outliers += 1

    loaded_to_subtract = np.array(loaded_to_subtract)

    # Crop the unused end of the array
    all_files_to_fit = all_files_to_fit[:index_to_insert]
    logger.info(f"Total of {len(all_files_to_fit)} images to fit, filling {all_files_to_fit.nbytes / 1024 ** 3:.2f} GB")
    logger.info(f"(Drawn from {len(input_cubes)} images to subtract and {len(files_to_fit)} extra images for fitting)")
    logger.info(f"({n_outliers} outliers were rejected from fitting)")


    bodies_in_quarter = np.array(bodies_in_quarter)

    return all_files_to_fit, bodies_in_quarter, loaded_to_subtract


def pca_filter_one_stride(stride: int, n_strides: int, bodies_in_quarter: np.ndarray, to_subtract: np.ndarray,
                          n_components: int, med_filt: int) -> tuple[np.ndarray, np.ndarray]:
    """Run PCA-based filtering for one stride position."""
    # This sets up a logger that forwards entries through a queue to the main process, where they can be forwarded to
    # Prefect
    logger = logging.getLogger()
    logger.addHandler(logging.handlers.QueueHandler(_log_queue))
    logger.setLevel(logging.INFO)

    quarter_slices = [
        np.s_[0:_all_files_to_fit.shape[1] // 2, 0:_all_files_to_fit.shape[2] // 2],
        np.s_[0:_all_files_to_fit.shape[1] // 2, _all_files_to_fit.shape[2] // 2:],
        np.s_[_all_files_to_fit.shape[1] // 2:, 0:_all_files_to_fit.shape[2] // 2],
        np.s_[_all_files_to_fit.shape[1] // 2:, _all_files_to_fit.shape[2] // 2:],
    ]

    stride_filter = np.arange(len(_all_files_to_fit)) % n_strides == stride
    # This will mark the images we'll be subtracting from---those are the only ones we'll drop from the fitting
    to_subtract_filter = stride_filter * (to_subtract >= 0)
    if not np.any(to_subtract_filter):
        logger.info(f"Stride {stride} has no images to subtract")
        return [], []

    images_to_subtract = _all_files_to_fit[to_subtract_filter]
    # This tracks where each image-to-be-subtracted is in the main list of NDCubes
    subtracted_cube_indices = to_subtract[to_subtract_filter]

    # We need to PCA separately for each quarter of the image
    filtered_by_quarter = np.empty_like(images_to_subtract)
    for i, quarter_slice in enumerate(quarter_slices):
        tag = f"stride {stride}, quarter {i+1}"
        logger.info(f"Starting to filter {tag}")
        # We mark the images that don't have any planets in the quarter we're filtering for (since those can
        # contaminate the PCA components)
        no_bodies_in_quarter = np.all(bodies_in_quarter[:, :, i] == False, axis=1) # noqa: E712
        images_to_fit = _all_files_to_fit[no_bodies_in_quarter * ~to_subtract_filter]
        filtered = run_pca_filtering(images_to_subtract, images_to_fit, n_components, med_filt, tag, logger)

        filtered_by_quarter[:, *quarter_slice] = filtered[:, *quarter_slice]

    return subtracted_cube_indices, filtered_by_quarter


def run_pca_filtering(images_to_subtract: np.ndarray, images_to_fit: np.ndarray, n_components: int,
                      med_filt: int, tag: str, logger: logging.Logger) -> np.ndarray:
    """Run PCA filtering."""
    pca = PCA(n_components=n_components)
    pca.fit(images_to_fit.reshape((len(images_to_fit), -1)))
    logger.info(f"Fitting finished for {tag}")

    transformed = pca.transform(images_to_subtract.reshape((len(images_to_subtract), -1)))

    if med_filt:
        for i in range(len(pca.components_)):
            comp = pca.components_[i].reshape(images_to_fit.shape[1:])
            comp = scipy.signal.medfilt2d(comp, med_filt)
            pca.components_[i] = comp.ravel()
        logger.info(f"Median smoothing finished for {tag}")

    reconstructed = pca.inverse_transform(transformed).reshape(images_to_subtract.shape)
    return images_to_subtract - reconstructed


def find_bodies_in_image(frame: str | NDCube | tuple[NormalizedMetadata, WCS]) -> list:
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
        frame, wcs = frame
        location = frame["GEOD_LON"].value, frame["GEOD_LAT"].value, frame["GEOD_ALT"].value
        image_shape = wcs.array_shape
    else:
        msg = "Type of 'frame' not recognized"
        raise TypeError(msg)

    results = []
    for body in ["Mercury", "Venus", "Moon", "Mars", "Jupiter", "Saturn"]:
        body_loc = get_body(body, time=Time(wcs.wcs.dateobs), location=EarthLocation.from_geodetic(*location))
        x, y = wcs.world_to_pixel(body_loc)
        w = 40 if body == "Moon" else 10
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
    # logging print statements), so in the workers we need to set up a logger than puts log entries in a Queue,
    # and then on the main process side we need to spawn a thread that monitors the queue and forwards log entries to
    # Prefect.

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
            pass
        _log_queue = None
