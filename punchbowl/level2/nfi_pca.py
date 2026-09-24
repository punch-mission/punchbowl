import os
import warnings
import multiprocessing as mp
from datetime import UTC, datetime
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor

import astropy
import numba
import numpy as np
import reproject
import scipy.signal
from astropy.coordinates import get_body
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from scipy.interpolate import RegularGridInterpolator
from skimage.restoration import inpaint_biharmonic
from sklearn.decomposition import PCA

from punchbowl.auto.control.util import batched
from punchbowl.data import NormalizedMetadata, get_base_file_name, load_ndcube_from_fits
from punchbowl.data.meta import check_moon_in_fov
from punchbowl.data.punchcube import PUNCHCube
from punchbowl.level1.dynamic_stray_light import phase_in_day
from punchbowl.prefect import get_logger, punch_task
from punchbowl.util import (
    ShmPickleableNDArray,
    limit_threads,
    load_mask_file,
    make_circular_mask,
    nan_percentile,
    nan_percentile_2d,
)


@punch_task
def pca_filter(input_files: list[str], context_files: list[str], nfi_mask: str, ref_date: str, n_components: int = 100,
               n_strides: int = 8, downsample_factor: int = 2, n_loaders: int = 4, n_workers: int = 20,
               ) -> list[PUNCHCube]:
    """
    Run PCA-based NFI filtering.

    There are several steps:
    1) The input images are loaded in parallel

    2) Downsampled copies of the images are made. Some steps are run at reduced resolution and the results are
    up-sampled back to the data.

    3) The images are searched for planets, the Moon, and saturated pixels. Copies of the images are made with these
    regions filled in smoothly. These copies also receive a spatial median filter.

    4) Images that are out-of-distribution are identified, so they can be excluded from the PCA fitting

    5) PCA components are fit to the downsampled, filled images from (3). (If outliers, planets, saturated pixels,
    etc. made it into this stage, we'd get artifacts in the PCA components.)

    6) Dynamic stray light models are generated from each image using the components and the downsampled,
    filled images. These models are upscaled and subtracted from the raw images. The models are also subtracted from
    the downsampled, filled images.

    7) Instrument-frame post-processing occurs. Each image receives something like an unsharp mask. (The downsampled,
    filled, PCA-filtered images are smoothed, upscaled, and subtracted from the full-res images. If planets,
    the moon, etc. made it into this smoothing, the subtraction would produce dark halos.) Each image then receives a
    spatial median filter. A low-percentile background across all images is then subtracted.

    8) The images are reprojected to the helio frame.

    9) To remove brightness variations that correlate with orbital phase, the images are fit with sinusoids whose
    frequencies are the orbital frequency, as well as a few harmonics. The sinusoids are then subtracted.

    10) The inner and outer edge of the data is trimmed, and the output cubes are assembled. We also write out the
    PCA components and the instrument-frame backgrounds, for use by QuickPUNCH.

    The output images from this flow are then put through the normal F corona modeling code to produce and subtract a
    helio-frame background.

    Because a large number of images are needed for outlier rejection, PCA fitting, background generation,
    and sinusoid fitting, this flow can accept "context" files---files which have already been through this flow,
    but which are made available to this flow to have a proper ensemble of images. These extra images run through all
    the steps, but output cubes are not generated for them.

    When PCA components are fit, we take a strided approach. The idea is that we'll be more confident we're not
    fitting and removing dynamic K corona structure if a given image is filtered using PCA components that have never
    seen that image. So we do N rounds of PCA filtering. For the first, we exclude from PCA fitting images N, 2N, 3N,
    etc., as well as the image before and after each of those. Those PCA components are used to subtract images N,
    2N, etc. The exclusions are then shifted relative to the image sequence and the process repeated, until we've
    done N rounds of PCA fitting. This lets each image be subtracted using components that have never seen that image
    (nor the images closest to it in time). To ensure this striding can be properly applied to QuickPUNCH images,
    we take this a little further and for each image, we compute which image number it is in the day's imaging
    sequence. The striding and exclusions are then based on that sequence number mod-N.

    Parameters
    ----------
    input_files : list[str]
        The files to be filtered
    context_files : list[str]
        The files to be used for PCA fitting but which don't need to be filtered
    nfi_mask : str
        Path to a NFI mask file
    ref_date : str
        The date to use for the PCA components
    n_components : int
        The number of PCA components to fit
    n_strides : int
        The number of strides to use
    downsample_factor : int
        How much the data should be down-sampled for PCA fitting
    n_loaders : int
        Number of worker processes for loading
    n_workers : int
        Number of worker processes for processing

    Returns
    -------
    output_data: list[PUNCHCube]
        The resulting data cubes

    """
    logger = get_logger()
    logger.info("Starting PCA flow")

    numba.set_num_threads(n_workers)
    context = mp.get_context("forkserver")
    with ProcessPoolExecutor(n_workers, mp_context=context) as process_pool:
        file_list = sorted(input_files + context_files)
        logger.info(f"Loading {len(input_files)} to filter and {len(context_files)} context files")
        x_cube, metas, wcses, cwcses, loaded_files, sat_mask_cube = load_files(file_list,
                                                                               n_workers=n_loaders,
                                                                               downsample_factor=downsample_factor)

        phases = np.array([phase_in_day(path) for path in loaded_files])

        nfi_mask = load_mask_file(nfi_mask)
        nfi_mask_ds = nfi_mask.reshape((x_cube.shape[1] // downsample_factor, downsample_factor,
                                        x_cube.shape[2] // downsample_factor, downsample_factor)).all(axis=(1, 3))

        x_cube_downsampled = ShmPickleableNDArray((x_cube.shape[0],
                                                   x_cube.shape[1] // downsample_factor,
                                                   x_cube.shape[1] // downsample_factor), dtype=x_cube.dtype)
        for i in range(len(x_cube)):
            x_cube_downsampled[i] = downsample(x_cube[i], downsample_factor)

        logger.info("Images loaded")

        x_cube_ds_filled, plot_masks = fill_problem_regions(x_cube_downsampled, metas, cwcses, sat_mask_cube,
                                                            downsample_factor, process_pool)

        x_cube_downsampled.free()
        del x_cube_downsampled

        logger.info("Problem regions filled")

        good_mask_headers = find_outliers_with_headers(metas)
        good_mask_pca = find_outliers_with_pca(x_cube_ds_filled, good_mask_headers, nfi_mask_ds, n_workers)
        good_mask = good_mask_headers * good_mask_pca

        dsl_models, pca_components = do_pca_filtering(x_cube_ds_filled, good_mask, nfi_mask_ds, phases, n_strides,
                                                      n_components, process_pool, n_workers)

        logger.info("PCA filtering complete")

        filtered_images, filtered_filled_images = subtract_models_from_data(x_cube, x_cube_ds_filled, dsl_models,
                                                                            downsample_factor, process_pool)

        x_cube.free()
        x_cube_ds_filled.free()
        dsl_models.free()
        del x_cube, x_cube_ds_filled, dsl_models

        logger.info("Components upsampled and subtracted")

        post_filtered_images, inst_frame_background = inst_frame_filter(
            filtered_images, filtered_filled_images, downsample_factor, process_pool)

        filtered_images.free()
        filtered_filled_images.free()
        del filtered_images, filtered_filled_images

        logger.info("Post-filtering complete")

        oriented_images, masks, target_frame = reproject_images(post_filtered_images, plot_masks, wcses, process_pool,
                                                                downsample_factor)

        post_filtered_images.free()
        plot_masks.free()
        del post_filtered_images, plot_masks

        logger.info("Images reprojected")

        corrected_frames = do_sinusoid_filtering(oriented_images, metas, nfi_mask, process_pool)

        oriented_images.free()
        del oriented_images

        logger.info("Sinusoidal trends removed")

        circular_mask = make_edge_mask(corrected_frames.shape[1:])
        corrected_frames *= circular_mask[None, :, :]

        output_cubes = []

        dates = [m.datetime for m in metas]
        new_meta = NormalizedMetadata.load_template("AR4", "1")
        new_meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        new_meta["DATE-OBS"] = ref_date
        new_meta["DATE-AVG"] = ref_date
        new_meta["DATE-BEG"] = min(dates).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        new_meta["DATE-END"] = max(dates).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        new_meta["PCANCOMP"] = n_components
        new_meta["PCADWNSP"] = downsample_factor
        new_meta["FILEVRSN"] = metas[0]["FILEVRSN"].value
        pca_cube = PUNCHCube(data=pca_components, meta=new_meta, wcs=target_frame)

        output_cubes.append(pca_cube)

        new_meta = NormalizedMetadata.load_template("SR4", "1")
        new_meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        new_meta["DATE-OBS"] = ref_date
        new_meta["DATE-AVG"] = ref_date
        new_meta["DATE-BEG"] = min(dates).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        new_meta["DATE-END"] = max(dates).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        new_meta["FILEVRSN"] = metas[0]["FILEVRSN"].value
        bg_cube = PUNCHCube(data=inst_frame_background * circular_mask, meta=new_meta, wcs=target_frame)
        output_cubes.append(bg_cube)

        for i, path in enumerate(loaded_files):
            if path in input_files:
                new_meta = NormalizedMetadata.load_template("CNN", "2")
                new_meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
                for key in metas[i]:
                    if ((key in ["DATE-OBS", "DATE-BEG", "DATE-AVG", "DATE-END", "FILEVRSN", "OUTLIER", "BADPKTS",
                                 "XACTTIME", "GEOD_LON", "GEOD_LAT", "GEOD_ALT", "LOS_ALT"]
                            or key[-4:] in ["_OBS", "_VOB"])
                            and key in new_meta):
                        new_meta[key] = metas[i][key].value
                _, _, _, _, moondist, xpix, ypix = check_moon_in_fov(
                    metas[i]["DATE-OBS"].value, wcs=wcses[i], image_shape=corrected_frames[i].shape)
                new_meta["MOONDIST"] = moondist[0]
                new_meta["MOON_X"] = xpix[0]
                new_meta["MOON_Y"] = ypix[0]
                new_meta["OUTLIER"] = not good_mask[i]
                new_meta["PCANCOMP"] = n_components
                new_meta["PCADWNSP"] = downsample_factor
                new_meta["PCACOMPS"] = get_base_file_name(pca_cube)
                new_meta["CALSL0"] = get_base_file_name(bg_cube)
                new_meta["CTRXNFI4"] = target_frame.wcs.crpix[1] - 1
                new_meta["CTRYNFI4"] = target_frame.wcs.crpix[0] - 1

                new_meta.provenance = [os.path.basename(path)]

                uncertainty = np.where(masks[i] < 0.5, np.inf, 1e-13)
                cube = PUNCHCube(data=corrected_frames[i], meta=new_meta, wcs=target_frame,
                                 uncertainty=StdDevUncertainty(uncertainty))
                output_cubes.append(cube)

        logger.info("PCA flow done!")
        return output_cubes


def reconstitute(flat_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Put selected data back into an array.

    If you did `my_pixels = my_array[mask]` to select certain pixels and operate on those pixels, this function takes
    `my_pixels` and puts them back where they were in an empty array of the original shape.

    Parameters
    ----------
    flat_image : np.ndarray
        The selected pixels. If 2D, it is assumed to be extractions from multiple images stacked along the first
        dimension.
    mask : np.ndarray
        The mask originally used to extract the pixels

    Returns
    -------
    np.ndarray
        The restored image(s)

    """
    if flat_image.ndim == 1:
        im = np.zeros(mask.shape)
        im[mask] = flat_image
        return im
    im = np.zeros((len(flat_image), *mask.shape))
    for i in range(len(flat_image)):
        im[i] = reconstitute(flat_image[i], mask)
    return im


def get_pylon_mask(shape: tuple, wcs: WCS, blur: bool = False) -> np.ndarray:
    """
    Generate a mask for the pylon region.

    Parameters
    ----------
    shape : tuple
        The mask shape
    wcs : WCS
        The WCS for the image we're masking
    blur : bool
        If true, the mask is blurred so it has a soft edge.

    Returns
    -------
    np.ndarray
        The mask

    """
    yy, xx = np.indices(shape, dtype=float)
    yy -= wcs.wcs.crpix[0] - 1 + 40
    xx -= wcs.wcs.crpix[1] - 1
    ang = np.arctan2(yy, xx) * 180 / np.pi
    mask = (np.abs(ang + 90) > 75) + (np.abs(xx) > 700)
    mask[(np.abs(ang + 90) < 65)] = 0
    if blur:
        mask = scipy.ndimage.gaussian_filter(mask.astype(np.float32), sigma=15)
    return mask


def _load_one_file(path: str, downsample_factor: int,
                   ) -> tuple[NormalizedMetadata, WCS, WCS, str, np.ndarray, np.ndarray] | str | None:
    """
    Load one file in a parallel worker.

    Parameters
    ----------
    path : str
        The path to load
    downsample_factor : int
        The factor by which the image data will later be downsampled.

    Returns
    -------
    meta : NormalizedMetadata
        The loaded metadata
    wcs : WCS
        The loaded helio wcs
    cwcs : WCS
        The loaded celestial wcs
    path : str
        The file path that was loaded
    data : np.ndarray
        The loaded image
    sat_mask_cube : np.ndarray
        The loaded saturation mask

    """
    if not os.path.exists(path):
        return "missing"
    cube = load_ndcube_from_fits(path, include_uncertainty=False, include_provenance=False, dtype=np.float32)
    if cube.meta["BADPKTS"].value or cube.meta["DATAP25"].value > 1e-9:
        return None

    data = cube.data
    saturation_mask = np.isinf(cube.uncertainty.array)
    if downsample_factor > 1:
        saturation_mask = saturation_mask.reshape((data.shape[0] // downsample_factor, downsample_factor,
                                                   data.shape[1] // downsample_factor, downsample_factor),
                                                  ).any(axis=(1, 3))
    return cube.meta, cube.wcs, cube.celestial_wcs, path, data, saturation_mask


def load_files(files: list[str], n_workers: int, downsample_factor: int,
               ) -> tuple[np.ndarray, list, list, list, list, np.ndarray]:
    """
    Load the input files.

    Loading is done in parallel. Terribly obvious outliers are skipped.

    Parameters
    ----------
    files : list[str]
        The files to load.
    n_workers : int
        The number of workers.
    downsample_factor : int
        The factor by which the saturation mask is downsampled. (Full-res data is returned because it's always needed,
        but if we're downsampling the data later, we'll only need the saturation mask at the reduced resolution.)

    Returns
    -------
    x_cube : np.ndarray
        The loaded data
    metas : list[NormalizedMetadata]
        The loaded metadata
    wcses : list[WCS]
        The loaded helio wcses
    cwcses : list[WCS]
        The loaded celestial wcses
    loaded_files : list[str]
        The file paths that were actually loaded (not skipped)
    sat_mask_cube : np.ndarray
        The saturation mask array

    """
    metas = []
    wcses = []
    cwcses = []
    loaded_files = []
    x_cube = ShmPickleableNDArray((len(files), 2048, 2048), dtype=np.float32)
    sat_mask_cube = ShmPickleableNDArray((len(files), 2048 // downsample_factor, 2048 // downsample_factor), dtype=bool)
    i = 0
    n_missing = 0
    context = mp.get_context("forkserver")
    with ProcessPoolExecutor(n_workers, mp_context=context) as process_pool:
        for result in process_pool.map(_load_one_file, files, repeat(downsample_factor), chunksize=2):
            # Files that were determined on load to be outliers got rejected
            if result is None:
                continue
            if result == "missing":
                n_missing += 1
                if n_missing > 0.05 * len(files):
                    raise RuntimeError("More than 5% of input files are missing")
                continue
            meta, wcs, cwcs, loaded_file, image, sat_mask = result
            metas.append(meta)
            wcses.append(wcs)
            cwcses.append(cwcs)
            loaded_files.append(loaded_file)
            x_cube[i] = image
            sat_mask_cube[i] = sat_mask
            i += 1
            # Each cube will have an identical distortion map, and we don't need to hold them all in memory
            wcs.cpdis1 = wcses[0].cpdis1
            wcs.cpdis2 = wcses[0].cpdis2
            cwcs.cpdis1 = wcses[0].cpdis1
            cwcs.cpdis2 = wcses[0].cpdis2
    x_cube = x_cube[:i]
    sat_mask_cube = sat_mask_cube[:i]
    return x_cube, metas, wcses, cwcses, loaded_files, sat_mask_cube


def _fill_one_image(src_data: np.ndarray, dest: np.ndarray, mask_dest: np.ndarray,
                    meta: NormalizedMetadata, wcs: WCS, sat_mask: np.ndarray,
                    downsample_factor: int) -> None:
    numba.set_num_threads(2)

    if downsample_factor > 1:
        wcs = wcs[::downsample_factor, ::downsample_factor]
    # Mark spots to in-paint
    fill_mask = np.zeros_like(src_data, dtype=bool)
    # Mark pixels that should be flagged in the output image
    plot_mask = np.zeros_like(fill_mask, dtype=np.float32)

    # We used to search for planets as well, but they seem to be handled well by the saturated-pixel flagging
    body_names = ["moon"]
    bodies = []
    for body in body_names:
        bodies.append(get_body(body, meta.astropy_time))
    bodies = bodies[0] if len(bodies) == 1 else astropy.coordinates.concatenate(bodies)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*failed to converge.*")
        warnings.filterwarnings("ignore", ".*All-NaN slice.*")

        xs, ys = wcs.world_to_pixel(bodies)

        for body, x, y in zip(body_names, np.atleast_1d(xs), np.atleast_1d(ys), strict=True):
            if 0 < x < src_data.shape[1] and 0 < y < src_data.shape[0]:
                x, y = int(x), int(y) # noqa: PLW2901
                w = 30 if body == "moon" else 9
                w = round(w * 2 / downsample_factor)
                fill_mask[y - w:y + w + 1, x - w:x + w + 1] = 1

    # Flag saturated pixels
    sat_idxs = np.nonzero(sat_mask)
    for y, x in zip(*sat_idxs, strict=True):
        fill_mask[y - 1:y + 2, x - 1:x + 2] = 1
        plot_mask[y - 1:y + 2, x - 1:x + 2] = 1

    # Every saturated pixel leaves a residual in the rest of its row, since the L1 de-streaking undercorrects. If
    # enough saturated pixels show up in a given row, the streak becomes visible in our images. So for rows with too
    # many saturated pixels, flag the whole row.
    for y in np.unique(sat_idxs[0]):
        if np.sum(sat_idxs[0] == y) > 9:
            fill_mask[y] = 1
            plot_mask[y - 1:y + 2] = 1

    # Generate replacement values for flagged pixels
    filtered_image = nan_percentile_2d(np.where(fill_mask, np.nan, src_data), 50, round(5 * 2 / downsample_factor))
    dest[:] = inpaint_biharmonic(filtered_image, fill_mask)
    mask_dest[:] = plot_mask


def fill_problem_regions(x_cube: np.ndarray, metas: list[NormalizedMetadata], cwcses: list[WCS],
                         sat_mask_cube: np.ndarray, downsample_factor: int,
                         process_pool: ProcessPoolExecutor) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate fill values for problem regions in images (e.g. the Moon or saturated pixels).

    Problem regions are identified and then filled with a smooth inpainting.

    Parameters
    ----------
    x_cube : np.ndarray
        The data to filter
    metas : list[NormalizedMetadata]
        Corresponding metadata
    cwcses : list[WCS]
        Corresponding wcses
    sat_mask_cube : np.ndarray
        Corresponding saturation masks
    downsample_factor: int
        The factor by which the provided data is downsampled
    process_pool : ProcessPoolExecutor
        Executor for parallel work

    Returns
    -------
    np.ndarray
        The filled data
    np.ndarray
        A mask cube indicating where fill values were generated

    """
    # Copy x_cube
    x_cube_filled = ShmPickleableNDArray.empty_like(x_cube)
    plot_masks = ShmPickleableNDArray.empty_like(x_cube)

    for _ in process_pool.map(_fill_one_image, x_cube, x_cube_filled, plot_masks, metas, cwcses, sat_mask_cube,
                              repeat(downsample_factor), chunksize=2):
        # Loop is necessary for any exceptions from workers to be raised
        pass

    return x_cube_filled, plot_masks


def find_outliers_with_pca(x_cube_filled: np.ndarray, good_mask: np.ndarray, nfi_mask: np.ndarray, n_workers: int,
                           ) -> np.ndarray:
    """
    Identify outliers with by calculating a few PCA components and finding images with very strong amplitudes.

    Parameters
    ----------
    x_cube_filled : np.ndarray
        The data to filter
    good_mask : np.ndarray
        A mask indicating whether images are good. Images already known to be bad will be excluded to make this test
        more sensitive.
    nfi_mask : np.ndarray
        The NFI mask
    n_workers : int
        Number of parallel threads to use

    Returns
    -------
    np.ndarray
        A mask indicating which images are good

    """
    data = x_cube_filled[:, nfi_mask]
    data = data[good_mask]

    with limit_threads(n_workers):
        pca_global = PCA(n_components=10)
        pca_global.fit(data)
        t_global = pca_global.transform(data)

    maybe_bad = set()
    for i in range(10):
        s = t_global[:, i]
        suspects = set(np.where(np.abs(s - np.median(s)) > 4 * np.std(s))[0])
        maybe_bad |= suspects

    indices = np.arange(len(x_cube_filled))
    bad_indices = indices[good_mask][list(maybe_bad)]
    good_mask = np.ones(len(x_cube_filled), dtype=bool)
    good_mask[bad_indices] = 0

    return good_mask


def find_outliers_with_headers(metas: list[NormalizedMetadata]) -> np.ndarray:
    """
    Identify outliers by checking stats in the image headers.

    A first round of outlier rejection. Images with percentiles exceeding the population median by two sigma are
    flagged.

    Parameters
    ----------
    metas : list[NormalizedMetadata]
        The image metadata

    Returns
    -------
    np.ndarray
        A mask indicating which images are good

    """
    good_mask = np.ones(len(metas), dtype=bool)
    for k in "DATAP98", "DATAP90", "DATAP50":
        d = np.array([m[k].value for m in metas])
        good_mask[d > np.median(d) + 2 * np.std(d)] = False
    return good_mask


def do_pca_filtering(x_cube_filled: np.ndarray, good_mask: np.ndarray, nfi_mask: np.ndarray, phases: np.ndarray,
                     n_strides: int, n_components: int, process_pool: ProcessPoolExecutor, n_workers: int,
                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute PCA components and corresponding dynamic stray light estimates.

    Parameters
    ----------
    x_cube_filled : np.ndarray
        The data to filter
    good_mask : np.ndarray
        Flag for each image indicating whether to include it when fitting components.
    nfi_mask : np.ndarray
        The NFI mask
    phases : np.ndarray
        The phase in the day of each NFI image. Used for reproducible striding.
    n_strides : int
        The number of stride positions to use
    n_components : int
        The number of components to fit
    process_pool : ProcessPoolExecutor
        An executor to use for parallelization
    n_workers : int
        Maximum number of cores to use. Will be divided among processes for each stride position and threads within
        each process.

    Returns
    -------
    np.ndarray
        The dynamic stray light models for each image
    np.ndarray
        The fitted PCA components

    """
    dsls = ShmPickleableNDArray.empty_like(x_cube_filled)
    pca_components = ShmPickleableNDArray((n_strides, n_components + 1, np.sum(nfi_mask)), dtype=np.float32)
    n_threads = max(1, round(n_workers / n_components))
    # Run once per stride position
    for _ in process_pool.map(_do_pca_filtering_one_stride, range(n_strides), repeat(n_strides), repeat(x_cube_filled),
                              repeat(good_mask), repeat(dsls), pca_components, repeat(nfi_mask), repeat(phases),
                              repeat(n_components), repeat(n_threads)):
        # Loop is necessary for any exceptions from workers to be raised
        pass
    return dsls, pca_components


def build_models_with_existing_components(
        x_cube_filled: np.ndarray, nfi_mask: np.ndarray, pca_components: np.ndarray,
        phases: np.ndarray, process_pool: ProcessPoolExecutor) -> np.ndarray:
    """
    Compute dynamic stray light estimates from previously-fitted PCA components.

    Parameters
    ----------
    x_cube_filled : np.ndarray
        The data to filter
    nfi_mask : np.ndarray
        The NFI mask
    pca_components : np.ndarray
        The PCA components to use
    phases : np.ndarray
        The phase in the day of each NFI image. Used for reproducible striding.
    process_pool : ProcessPoolExecutor
        An executor to use for parallelization

    Returns
    -------
    np.ndarray
        The dynamic stray light models for each image

    """
    dsls = ShmPickleableNDArray.empty_like(x_cube_filled)

    smoothed_components = ShmPickleableNDArray.empty_like(pca_components)
    comp_smoothing = 5
    for i in range(pca_components.shape[0]):
        for j in range(pca_components.shape[1]):
            component = reconstitute(pca_components[i, j], nfi_mask)
            component = scipy.signal.medfilt2d(component, comp_smoothing)
            smoothed_components[i, j] = component[nfi_mask]

    for _ in process_pool.map(_make_one_model_from_components, phases, repeat(pca_components),
                              repeat(smoothed_components), dsls, repeat(nfi_mask)):
        # Loop is necessary for any exceptions from workers to be raised
        pass
    return dsls


def _make_one_model_from_components(image: np.ndarray, phase: int, pca_components: np.ndarray,
                                    smoothed_pca_components: np.ndarray, dsl_dest: np.ndarray, nfi_mask: np.ndarray,
                                    ) -> None:
    n_components = pca_components.shape[1]
    means = pca_components[phase][0]
    components = pca_components[phase][1:]
    smoothed_components = smoothed_pca_components[phase][1:]

    pca = PCA(n_components=n_components)
    pca.mean_ = means
    pca.components_ = components
    # These values don't matter but must be present
    pca.explained_variance_ = np.ones(n_components)
    # This function is being run in parallel
    with limit_threads(1):
        t = pca.transform(image[nfi_mask].reshape((1, -1)))
        pca.components_ = smoothed_components
        recon = pca.inverse_transform(t)

    dsl_dest[:] = reconstitute(recon, nfi_mask)


def _do_pca_filtering_one_stride(this_set_number: int, n_sets: int, x_cube_filled: np.ndarray, good_mask: np.ndarray,
                                 dsl_dest: np.ndarray, component_dest: np.ndarray, nfi_mask: np.ndarray,
                                 phases: np.ndarray, n_components: int, n_threads: int) -> None:
    phases = (phases - this_set_number) % n_sets
    fit_idxs = (phases != 0) * (phases != 1) * (phases != n_sets - 1) * good_mask
    set_to_fit = x_cube_filled[fit_idxs][:, nfi_mask]

    filter_idxs = phases == 0
    set_to_filter = x_cube_filled[filter_idxs][:, nfi_mask]

    pca = PCA(n_components=n_components)
    with limit_threads(n_threads):
        pca.fit(set_to_fit)
        component_dest[0] = pca.mean_
        component_dest[1:] = pca.components_

        t = pca.transform(set_to_filter)

        comp_smoothing = 5
        for j in range(len(pca.components_)):
            component = reconstitute(pca.components_[j], nfi_mask)
            component = scipy.signal.medfilt2d(component, comp_smoothing)
            pca.components_[j] = component[nfi_mask]

        recon = pca.inverse_transform(t)

    dest_idxs = np.arange(len(x_cube_filled))[filter_idxs]
    dsl_dest[dest_idxs] = reconstitute(recon, nfi_mask)


def subtract_models_from_data(x_cube: np.ndarray, x_cube_ds_filled: np.ndarray, dsl_models: np.ndarray,
                              downsample_factor: int, process_pool: ProcessPoolExecutor,
                              ) -> tuple[np.ndarray, np.ndarray]:
    """
    Upscale and subtract dynamic stray light models from data.

    Parameters
    ----------
    x_cube : np.ndarray
        The full-resolution data from which to subtract the models
    x_cube_ds_filled : np.ndarray
        The downsampled, inpainted data from which to subtract the models
    dsl_models : np.ndarray
        The dynamic stray light models
    downsample_factor : int
        The previously-used downsampling factor--we'll upscale by this amount.
    process_pool : ProcessPoolExecutor
        A ProcessPoolExecutor to use for multiprocessing

    Returns
    -------
    np.ndarray
        The full-resolution, model-subtracted data
    np.ndarray
        The downsampled and filtered, model-subtracted data

    """
    filtered_images = ShmPickleableNDArray.empty_like(x_cube)
    filtered_filled_images = ShmPickleableNDArray.empty_like(x_cube_ds_filled)
    for _ in process_pool.map(_subtract_one_model_from_data, x_cube, x_cube_ds_filled, dsl_models,
                              filtered_images, filtered_filled_images, repeat(downsample_factor), chunksize=2):
        # Loop is necessary for any exceptions from workers to be raised
        pass
    return filtered_images, filtered_filled_images


def _subtract_one_model_from_data(x_data: np.ndarray, x_data_ds_filled: np.ndarray, dsl_model: np.ndarray,
                                  filtered_image: np.ndarray, filtered_filled_image:np.ndarray,
                                  downsample_factor: int) -> None:
    upsampled_model = upsample(dsl_model, downsample_factor)
    np.subtract(x_data, upsampled_model, out=filtered_image)
    np.subtract(x_data_ds_filled, dsl_model, out=filtered_filled_image)


def upsample(image: np.ndarray, factor: int) -> np.ndarray:
    """
    Upsample an image.

    Tries to be very careful about being an exact inverse of binning. Each pixel in the downsampled image is
    positioned at the center of the corresponding bin in the full-res image, and the data are then interpolated to
    full-res. Padded with zero at the edges where interpolation can't happen.

    Parameters
    ----------
    image : np.ndarray
        The image to upsample
    factor : int
        The factor to upsample by

    Returns
    -------
    np.ndarray
        The upsampled image

    """
    if factor == 1:
        return image
    target_shape = np.array(image.shape) * factor
    binned_indices = (np.arange(0.5 * (factor - 1), target_shape[0], factor),
                      np.arange(0.5 * (factor - 1), target_shape[1], factor))
    upsample_indices = np.stack(np.indices(target_shape), axis=-1)
    interp = RegularGridInterpolator(binned_indices, image, method="linear", bounds_error=False, fill_value=0)
    return interp(upsample_indices)


def downsample(image: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample an image by binning and averaging.

    Parameters
    ----------
    image : np.ndarray
        The image to downsample
    factor : int
        The factor to downsample by

    Returns
    -------
    np.ndarray
        The downsampled image

    """
    if factor == 1:
        return image
    if len(image.shape) == 2:
        return image.reshape((image.shape[0] // factor, factor,
                              image.shape[1] // factor, factor)).mean(axis=(1, 3))
    return image.reshape((image.shape[0],
                          image.shape[1] // factor, factor,
                          image.shape[2] // factor, factor)).mean(axis=(2, 4))


def _inst_frame_filter_one_image(image: np.ndarray, filled_image: np.ndarray,
                                 destination: np.ndarray, downsample_factor: int) -> None:
    ims1 = scipy.ndimage.percentile_filter(filled_image, 30, size=round(11 * 2 / downsample_factor))
    ims2 = scipy.ndimage.gaussian_filter(filled_image, sigma=15 * 2 / downsample_factor)
    unsharp_image = 0.4 * ims1 + 0.35 * ims2
    unsharp_image = upsample(unsharp_image, downsample_factor)
    unsharp_masked = image - unsharp_image
    destination[:] = scipy.ndimage.percentile_filter(unsharp_masked, 50, size=5)


def inst_frame_filter(filtered_images: np.ndarray, filtered_filled_images: np.ndarray,
                      downsample_factor: int, process_pool: ProcessPoolExecutor,
                      background_image: np.ndarray = None) -> np.ndarray:
    """
    Apply a spatial post-filter to the PCA-filtered images in the instrument frame.

    The filter is mostly just unsharp-masking followed by a spatial median filter, and then a low-percentile
    background across the ensemble.

    Parameters
    ----------
    filtered_images : np.ndarray
        The PCA-filtered images to post-filter
    filtered_filled_images : np.ndarray
        The PCA-filtered images with problem regions filled, used to build the unsharp-mask background.
    downsample_factor : int
        The factor by which `filtered_filled_images` is downsampled. The unsharp mask will be upscaled by this factor.
    process_pool : ProcessPoolExecutor
        A ProcessPoolExecutor to use for multiprocessing
    background_image : np.ndarray
        An optional image-frame background to subtract. If not provided, a percentile is calculated and subtracted.

    Returns
    -------
    np.ndarray
        The post-filtered images
    np.ndarray
        The instrument-frame background image that was computed and subtracted

    """
    post_filtered_images = ShmPickleableNDArray.empty_like(filtered_images)
    for _ in process_pool.map(_inst_frame_filter_one_image, filtered_images, filtered_filled_images,
                              post_filtered_images, repeat(downsample_factor), chunksize=2):
        # Loop is necessary for any exceptions from workers to be raised
        pass

    if background_image is None:
        background_image = nan_percentile(post_filtered_images, 4)
    post_filtered_images -= background_image

    return post_filtered_images, background_image


def censor_wcs(wcs: WCS) -> WCS:
    """
    Remove observer details from a WCS.

    When input images have slightly different viewpoints, Sunpy will say this
    is an invalid coordinate transformation. Here we censor information from the
    WCS to pacify Sunpy.

    Parameters
    ----------
    wcs : WCS
        The WCS to censor

    Returns
    -------
    WCS
        The censored WCS

    """
    wcs = wcs.deepcopy()
    wcs.wcs.aux.hgln_obs = None
    wcs.wcs.aux.hglt_obs = None
    wcs.wcs.aux.dsun_obs = None
    wcs.wcs.dateobs = ""
    wcs.wcs.dateavg = ""
    wcs.wcs.datebeg = ""
    wcs.wcs.dateend = ""
    return wcs


def _reproject_one_image(image: np.ndarray, plot_mask: np.ndarray, wcs: WCS, mask: np.ndarray, target_frame: WCS,
                         image_dest: np.ndarray, mask_dest: np.ndarray, downsample_factor: int) -> None:
    plot_mask = upsample(plot_mask, downsample_factor)
    mask = mask * (plot_mask == 0)
    input_data = np.stack((image, mask), dtype=np.float32)
    # Pre-create the output array so we can ensure it's 32-bit
    output_data = np.empty_like(input_data)
    reproject.reproject_adaptive((input_data, censor_wcs(wcs)), censor_wcs(target_frame), output_array=output_data,
                                 roundtrip_coords=False, return_footprint=False)
    image, mask = output_data
    np.nan_to_num(image, copy=False)
    np.nan_to_num(mask, copy=False)
    mask = mask > 0.5

    image_dest[:] = image
    mask_dest[:] = mask


def reproject_images(dfiltered_images: np.ndarray, plot_masks: np.ndarray, wcses: list[WCS],
                     process_pool: ProcessPoolExecutor, downsample_factor: int) -> tuple[np.ndarray, np.ndarray, WCS]:
    """
    Reproject images to the helio frame.

    Parameters
    ----------
    dfiltered_images : np.ndarray
        The doubly-filtered images to reproject
    plot_masks : np.ndarray
        Masks to take along through the reprojection
    wcses : list[WCS]
        The image WCSes
    process_pool : ProcessPoolExecutor
        The ProcessPoolExecutor to use for multiprocessing
    downsample_factor : int
        The factor by which the masks have been downsampled. They will be upsampled before reprojection

    Returns
    -------
    np.ndarray
        The reprojected images
    np.ndarray
        The reprojected masks
    WCS
        The frame the data were reprojected into

    """
    # We just need a frame that's north-up. Take a single NFI WCS, strip out the unique pointing and rotation info,
    # but keep the projection.
    target_frame = wcses[0].deepcopy()
    target_frame.wcs.pc = np.eye(2)
    target_frame.wcs.crpix = dfiltered_images.shape[2] / 2 + 0.5, dfiltered_images.shape[1] / 2 + 0.5
    target_frame.wcs.crval = 0, 0
    target_frame.cpdis1 = None
    target_frame.cpdis2 = None

    # We'll take this mask for the pylon region and reproject it with each image, so we can flag the region
    pylon_mask = get_pylon_mask(dfiltered_images[0].shape, wcses[0])

    oriented_images = ShmPickleableNDArray.empty_like(dfiltered_images)
    masks = ShmPickleableNDArray(dfiltered_images.shape, dtype=bool)
    for _ in process_pool.map(_reproject_one_image, dfiltered_images, plot_masks, wcses, repeat(pylon_mask),
                              repeat(target_frame), oriented_images, masks, repeat(downsample_factor)):
        # Loop is necessary for any exceptions from workers to be raised
        pass
    return oriented_images, masks, target_frame


def sinusoid(x: int | float | np.ndarray, dy: np.ndarray, *args: list[float]) -> np.ndarray:
    """
    Compute a sum of sinusoids.

    Each additional sinusoid is a higher harmonic of the first, which has a frequency of 1/360 degrees.

    Parameters
    ----------
    x : int | float | np.ndarray
        The points at which to compute the sinusoids
    dy : np.ndarray
        A zero-frequency component
    args : list[float]
        The first and second values are the amplitude and phase offset of the first sinusoid. Each additional two values
        are the same for the next sinusoid.

    Returns
    -------
    np.ndarray
        The sum of the sinusoids and fixed offset

    """
    result = dy if isinstance(x, (int, float)) else np.full(len(x), dy, dtype=float)
    for f, (A, dphi) in enumerate(batched(args, 2)): # noqa: N806
        result += A * np.sin(x * ((f + 1) * np.pi / 180) + dphi)
    return result


def jac(x: np.ndarray, dy: np.ndarray, *args: list[float]) -> np.ndarray: # noqa: ARG001
    """
    Compute the jacobian of `sinusoid`.

    Parameters
    ----------
    x : int | float | np.ndarray
        The points at which to compute the sinusoids
    dy : np.ndarray
        A zero-frequency component
    args : list[float]
        The first and second values are the amplitude and phase offset of the first sinusoid. Each additional two values
        are the same for the next sinusoid.

    Returns
    -------
    np.ndarray
        The Jacobian

    """
    ddy = np.full_like(x, 1)
    ret = [ddy]
    for f, (A, dphi) in enumerate(batched(args, 2)): # noqa: N806
        ret.append(np.sin(x * ((f + 1) * np.pi / 180) + dphi))
        ret.append(A * np.cos(x * ((f + 1) * np.pi / 180) + dphi))

    return np.transpose(ret)


def make_correction_map(crota: float, popts: np.ndarray, ivals: np.ndarray) -> np.ndarray:
    """
    Compute a 2D correction map for a given image from a grid of fit sinusoids.

    Parameters
    ----------
    crota : float
        The orbital phase value for which to prepare a map
    popts : np.ndarray
        The fitted sinusoid components
    ivals : np.ndarray
        The grid points at which the sinusoids were computed.

    Returns
    -------
    A 2D map containing at each pixel the value of the sinusoids for that pixel at the given CROTA value

    """
    correction_map = np.zeros(popts.shape[1:])

    for i in range(len(ivals)):
        for j in range(len(ivals)):
            correction_map[i, j] = sinusoid(crota, *popts[:, i, j])
    return correction_map


def _do_one_fit(args: tuple, crota_vals: np.ndarray, images: np.ndarray, mask: np.ndarray, n_comps: int, whs: int,
                ivals: np.ndarray) -> list:
    ix, i = args
    rets = []
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*Degrees of freedom.*")
        warnings.filterwarnings("ignore", ".*Mean of empty slice.*")
        for jx, j in enumerate(ivals):
            if np.any(mask[i - whs:i + whs + 1, j - whs:j + whs + 1] == 0):
                continue
            # Take the pixels in a small window around our grid point, to be our y values to fit
            x, y = crota_vals, images[:, i - whs:i + whs + 1, j - whs:j + whs + 1]
            x = np.broadcast_to(x, y.T.shape).T.ravel()
            y = y.ravel()

            # Find outliers
            med = np.median(y)
            bad = np.abs(y - med) / np.std(y) > 2
            x = x[~bad]
            y = y[~bad]

            p0 = [1e12 * np.median(y)] + [1, 0] * n_comps
            lbounds = [-np.inf] + [0, -np.inf] * n_comps
            ubounds = [np.inf] + [np.inf, np.inf] * n_comps

            if not len(y):
                continue
            try:
                with limit_threads(2), np.errstate(all="ignore"), warnings.catch_warnings():
                    warnings.filterwarnings("ignore", ".*Mean of empty slice.*")
                    warnings.filterwarnings("ignore", ".*Degrees of freedom <= 0.*")

                    # The y values are scaled up to ~1-ish, to play nicely with the termination thresholds
                    popt, _ = scipy.optimize.curve_fit(sinusoid, x, y * 1e12, p0=p0, jac=jac,
                                                       bounds=[lbounds, ubounds],
                                                       loss="cauchy", f_scale=0.5,
                                                       method="trf",
                                                       )
            except RuntimeError:
                popt = np.full(n_comps * 2 + 1, np.nan)
            # We need to fit the center to meaningfully fit the sinusoids, but we don't want to subtract out that
            # center. (The "f corona" subtraction step will handle that)
            popt[0] = 0
            # Scale amplitudes back down to MSB units
            for n in range(1, len(popt), 2):
                popt[n] *= 1e-12
            rets.append((popt, ix, jx))
    return rets


def compute_popts(crota_vals: np.ndarray, images: np.ndarray, mask: np.ndarray, process_pool: ProcessPoolExecutor,
                  n_comps: int, grid_size: int = 110, whs: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit sinusoids for one time window at a grid of pixel positions.

    Parameters
    ----------
    crota_vals : np.ndarray
        The orbital phase values---the x axis of the sinusoids
    images : np.ndarray
        The images to fit
    mask : np.ndarray
        The NFI mask
    process_pool : ProcessPoolExecutor
        A ProcessPoolExecutor to use for parallelization
    n_comps : int
        The number of sinusoids to fit. Each additional component will be a higher-yet harmonic.
    grid_size : int
        The number of grid points in each dimension at which to compute fits
    whs : int
        The window half-size---how big of a box around each grid point to use as the y-values for the fit.

    Returns
    -------
    np.ndarray
        The fitted coefficients, shape (n_comps, grid_size_y, grid_size_x)
    np.ndarray
        The pixel locations of the grid points. This is a one-dimensional array, and these values are used for both the
        x and y dimension.

    """
    ivals = np.round(np.linspace(0, images.shape[1], grid_size)).astype(int)
    popts = np.zeros((2 * n_comps + 1, ivals.size, ivals.size))

    for ret in process_pool.map(_do_one_fit, enumerate(ivals), repeat(crota_vals), repeat(images), repeat(mask),
                                repeat(n_comps), repeat(whs), repeat(ivals)):
        for popt, ix, jx in ret:
            popts[:, ix, jx] = popt
    return popts, ivals


def _desinusoid_one_image(src_image: np.ndarray, crota: float, t: float, ivals: np.ndarray,
                          time_based_popts: np.ndarray, dest: np.ndarray) -> None:
    if t <= time_based_popts[0][0]:
        correction_map = make_correction_map(crota, time_based_popts[0][1], ivals)
    elif t >= time_based_popts[-1][0]:
        correction_map = make_correction_map(crota, time_based_popts[-1][1], ivals)
    else:
        # Interpolate in time between two maps
        i = 0
        while not time_based_popts[i][0] <= t <= time_based_popts[i + 1][0]:
            i += 1
        t1 = time_based_popts[i][0]
        t2 = time_based_popts[i + 1][0]
        map1 = make_correction_map(crota, time_based_popts[i][1], ivals)
        map2 = make_correction_map(crota, time_based_popts[i + 1][1], ivals)
        correction_map = (t - t1) / (t2 - t1) * (map2 - map1) + map1

    correction_map = scipy.ndimage.median_filter(correction_map, 9)
    interp = RegularGridInterpolator([ivals] * 2, correction_map, bounds_error=False, fill_value=None)
    correction = interp(np.stack(np.mgrid[:src_image.shape[0], :src_image.shape[1]], axis=-1))
    dest[:] = src_image - correction


def do_sinusoid_filtering(oriented_images: np.ndarray, metas: list[NormalizedMetadata], mask: np.ndarray,
                          process_pool: ProcessPoolExecutor) -> np.ndarray:
    """
    Run orbital-phase de-trending.

    At a grid of pixel locations, a few sinusoids are fit to the time-series of images as a function of orbital
    phase. The sinusoid frequencies are the orbital frequency plus a few harmonics. The resulting sinusoids are then
    subtracted from the data. The image sequence is broken up into 3-day windows, and each window is fit separately
    (assuming sufficient samples in the window). To subtract a given image, the sinusoid windows are interpolated
    between.

    Parameters
    ----------
    oriented_images : np.ndarray
        The images to fit and filter
    metas : list[NormalizedMetadata]
        The image metadata
    mask : np.ndarray
        The NFI mask
    process_pool : ProcessPoolExecutor
        A ProcessPoolExecutor to use for parallelization

    Returns
    -------
    np.ndarray
        The filtered images

    """
    crota_vals = np.array([m["CROTA"].value for m in metas])
    time_based_popts = []
    dateobses = np.array([m["DATE-OBS"].value for m in metas], dtype=np.datetime64)

    # Divide the data into windows in time, to be fit separately
    t0 = dateobses[0]
    dt = np.timedelta64("3", "D")
    while t0 < dateobses[-1]:
        cut = (dateobses > t0) * (dateobses < t0 + dt)
        # Don't fit windows with too few images
        if np.sum(cut) > 250:
            idxs = np.nonzero(cut)[0]
            istart, istop = idxs[0], idxs[-1] + 1
            tmid = t0 + dt / 2
            popt, ivals = compute_popts(crota_vals[istart:istop], oriented_images[istart:istop], mask, process_pool,
                                        n_comps=2)
            time_based_popts.append((tmid, popt))
        t0 += dt

    if time_based_popts:
        corrected_frames = ShmPickleableNDArray.empty_like(oriented_images)
        for _ in process_pool.map(_desinusoid_one_image, oriented_images, crota_vals, dateobses, repeat(ivals),
                                  repeat(time_based_popts), corrected_frames):
            # Loop is necessary for any exceptions from workers to be raised
            pass
    else:
        corrected_frames = oriented_images

    return corrected_frames


def make_edge_mask(shape: tuple) -> np.ndarray:
    """
    Make a mask to trim the inner and outer edges of NFI.

    Parameters
    ----------
    shape : tuple
        The image shape

    Returns
    -------
    np.ndarray
        A mask with an inner and outer edge.

    """
    inner_mask = ~make_circular_mask(shape, 200)
    outer_mask = make_circular_mask(shape, 960)
    return inner_mask * outer_mask
