from collections.abc import Callable

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

from punchbowl.prefect import punch_task
from punchbowl.util import load_image_task


@punch_task()
def pca_filter(input_cube: NDCube, files_to_fit: list[NDCube | Callable | str],
               n_components: int=50, med_filt: int=5) -> None:
    """Run PCA-based filtering."""
    logger = get_run_logger()

    all_files_to_fit, bodies_in_quarter = load_files(input_cube, files_to_fit)

    quarter_slices = [
        np.s_[0:all_files_to_fit.shape[1] // 2, 0:all_files_to_fit.shape[2] // 2],
        np.s_[0:all_files_to_fit.shape[1] // 2, all_files_to_fit.shape[2] // 2:],
        np.s_[all_files_to_fit.shape[1] // 2:, 0:all_files_to_fit.shape[2] // 2],
        np.s_[all_files_to_fit.shape[1] // 2:, all_files_to_fit.shape[2] // 2:],
    ]

    filtered_by_quarter = np.empty_like(input_cube.data)
    for i, quarter_slice in enumerate(quarter_slices):
        logger.info(f"Starting to filter quarter {i+1}")
        no_bodies_in_quarter = np.all(bodies_in_quarter[:, :, i] == False, axis=1) # noqa: E712
        images_for_quarter = all_files_to_fit[no_bodies_in_quarter]
        filtered = run_pca_filtering(input_cube, images_for_quarter, n_components, med_filt)

        filtered_by_quarter[quarter_slice] = filtered[quarter_slice]

    input_cube.data[...] = filtered_by_quarter
    logger.info("PCA filtering finished")


def check_file(mean: float, median: float) -> bool:
    """Check if a file should be used."""
    if not (.3e-10 < median < 1.2e-10):
        return False
    if mean > 1.2e-10: # noqa: SIM103
        return False
    return True


@punch_task
def load_files(input_cube: NDCube, files_to_fit: list[NDCube | str | Callable]) -> np.ndarray:
    """Load files."""
    logger = get_run_logger()

    # We'll pre-allocate an array to load files into. Since we'll reject any bad files, we'll track an insertion
    # index as we add good files, and at the end we'll slice the array to drop any empty spots at the end. When
    # np.empty allocates memory, the OS doesn't *actually* allocate those pages until they're used, so we won't
    # actually use any more RAM than needed.
    all_files_to_fit = np.empty((len(files_to_fit), *input_cube.data.shape), dtype=input_cube.data.dtype)
    index_to_insert = 0
    bodies_in_quarter = []
    for input_file in files_to_fit:
        if isinstance(input_file, NDCube):
            if check_file(input_file.meta["DATAAVG"].value, input_file.meta["DATAMDN"].value):
                all_files_to_fit[index_to_insert] = input_file.data
                index_to_insert += 1
                bodies_in_quarter.append(find_bodies_in_image(input_file))
        elif isinstance(input_file, str):
            cube = load_image_task(input_file, include_provenance=False, include_uncertainty=False)
            if check_file(cube.meta["DATAAVG"].value, cube.meta["DATAMDN"].value):
                all_files_to_fit[index_to_insert] = cube.data
                index_to_insert += 1
                bodies_in_quarter.append(find_bodies_in_image(cube))
        elif isinstance(input_file, Callable):
            mean, median, bodies = input_file(all_files_to_fit[index_to_insert])
            if check_file(mean, median):
                # This is a bad file. Don't increment the insertion index, so this one gets overwritten by the next file
                index_to_insert += 1
                bodies_in_quarter.append(bodies)
        else:
            raise TypeError(f"Invalid type {type(input_file)} for input file")

    # Crop the unused end of the array
    all_files_to_fit = all_files_to_fit[:index_to_insert]
    logger.info(
        f"Loaded {len(files_to_fit)} images to fit")
    logger.info(f"Kept {index_to_insert}, filling {all_files_to_fit.nbytes / 1024 ** 3:.2f} GB")

    bodies_in_quarter = np.array(bodies_in_quarter)

    return all_files_to_fit, bodies_in_quarter


@punch_task
def run_pca_filtering(input_cube: NDCube, all_files_to_fit: np.ndarray, n_components: int, med_filt: int) -> np.ndarray:
    """Run PCA filtering."""
    logger = get_run_logger()
    with threadpool_limits(30):
        pca = PCA(n_components=n_components)
        pca.fit(all_files_to_fit.reshape((len(all_files_to_fit), -1)))
        logger.info("Fitting finished")

        transformed = pca.transform(input_cube.data.reshape((1, -1)))

        if med_filt:
            for i in range(len(pca.components_)):
                comp = pca.components_[i].reshape(input_cube.data.shape)
                comp = scipy.signal.medfilt2d(comp, med_filt)
                pca.components_[i] = comp.ravel()
            logger.info("Median smoothing finished")

        reconstructed = pca.inverse_transform(transformed).reshape(input_cube.data.shape)
        return input_cube.data - reconstructed


def find_bodies_in_image(frame: str | NDCube | WCS) -> list:
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
    elif not isinstance(frame, WCS):
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
