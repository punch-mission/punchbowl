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
from punchbowl.data import NormalizedMetadata, load_ndcube_from_fits
from punchbowl.data.meta import check_moon_in_fov
from punchbowl.data.punchcube import PUNCHCube
from punchbowl.prefect import get_logger, punch_task
from punchbowl.util import ShmPickleableNDArray, limit_threads, load_mask_file, nan_percentile, nan_percentile_2d


@punch_task
def pca_filter(input_files: list[str], context_files: list[str], nfi_mask: str, n_components: int = 100,
               n_strides: int = 8, downsample_factor_factor: int = 2, n_loaders: int = 4, n_workers: int = 20) -> list[PUNCHCube]:
    """Run PCA-based filtering."""
    logger = get_logger()
    logger.info("Starting PCA flow")

    numba.set_num_threads(n_workers)
    context = mp.get_context("forkserver")
    with ProcessPoolExecutor(n_workers, mp_context=context) as process_pool:
        file_list = sorted(input_files + context_files)
        x_cube, metas, wcses, cwcses, loaded_files, sat_mask_cube = load_files(file_list,
                                                                               n_workers=n_loaders,
                                                                               downsample_factor=downsample_factor_factor)

        nfi_mask = load_mask_file(nfi_mask)
        nfi_mask_ds = nfi_mask.reshape((x_cube.shape[1] // downsample_factor_factor, downsample_factor_factor,
                                        x_cube.shape[2] // downsample_factor_factor, downsample_factor_factor)).all(axis=(1, 3))

        x_cube_downsampled = ShmPickleableNDArray((x_cube.shape[0],
                                                   x_cube.shape[1] // downsample_factor_factor,
                                                   x_cube.shape[1] // downsample_factor_factor), dtype=x_cube.dtype)
        for i in range(len(x_cube)):
            x_cube_downsampled[i] = downsample(x_cube[i], downsample_factor_factor)

        logger.info("Images loaded")

        x_cube_ds_filled, plot_masks = fill_problem_regions(x_cube_downsampled, metas, cwcses, sat_mask_cube,
                                                            nfi_mask_ds, downsample_factor_factor, process_pool)

        x_cube_downsampled.free()
        del x_cube_downsampled

        logger.info("Problem regions filled")

        good_mask_headers = find_outliers_with_headers(metas)
        good_mask_pca = find_outliers_with_PCA(x_cube_ds_filled, good_mask_headers, nfi_mask_ds, n_workers)
        good_mask = good_mask_headers * good_mask_pca

        dsl_models = do_PCA_filtering(x_cube_ds_filled, good_mask, nfi_mask_ds, n_strides, n_components, process_pool,
                                      n_workers)

        logger.info("PCA filtering complete")

        filtered_images, filtered_filled_images = subtract_models_from_data(x_cube, x_cube_ds_filled, dsl_models,
                                                                            downsample_factor_factor, process_pool)

        x_cube.free()
        x_cube_ds_filled.free()
        dsl_models.free()
        del x_cube, x_cube_ds_filled, dsl_models

        logger.info("Components upsampled and subtracted")

        post_filtered_images = inst_frame_filter(filtered_images, filtered_filled_images, downsample_factor_factor, process_pool)

        filtered_images.free()
        filtered_filled_images.free()
        del filtered_images, filtered_filled_images

        logger.info("Post-filtering complete")

        oriented_images, masks, target_frame = reproject_images(post_filtered_images, plot_masks, wcses, process_pool,
                                                                downsample_factor_factor)

        post_filtered_images.free()
        plot_masks.free()
        del post_filtered_images, plot_masks

        logger.info("Images reprojected")

        corrected_frames = do_sinusoid_filtering(oriented_images, metas, nfi_mask, process_pool)

        oriented_images.free()
        del oriented_images

        logger.info("Sinusoidal trends removed")

        yy, xx = np.mgrid[:corrected_frames.shape[1], :corrected_frames.shape[2]]
        xx = xx - corrected_frames.shape[2] / 2 + 0.5
        yy = yy - corrected_frames.shape[1] / 2 + 0.5
        r = np.sqrt(xx ** 2 + yy ** 2)
        inner_mask = r > 200
        outer_mask = r < 960
        circular_mask = inner_mask * outer_mask

        output_cubes = []
        for i, path in enumerate(file_list):
            if path in input_files:
                new_meta = NormalizedMetadata.load_template("CNN", "2")
                new_meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
                for key in metas[i].keys():
                    if (key in ["DATE-OBS", "DATE-BEG", "DATE-AVG", "DATE-END", "FILEVRSN", "OUTLIER", "BADPKTS",
                                "XACTTIME", "GEOD_LON", "GEOD_LAT", "GEOD_ALT", "LOS_ALT"]
                            or key[-4:] in ["_OBS", "_VOB"]):
                        new_meta[key] = metas[i][key].value
                _, _, _, _, moondist, xpix, ypix = check_moon_in_fov(
                    metas[i]["DATE-OBS"].value, wcs=wcses[i], image_shape=corrected_frames[i].shape)
                new_meta["MOONDIST"] = moondist[0]
                new_meta["MOON_X"] = xpix[0]
                new_meta["MOON_Y"] = ypix[0]

                new_meta.provenance = [os.path.basename(path)]

                uncertainty = np.full(corrected_frames[i].shape, 1e-13)
                uncertainty[masks[i] < 0.5] = np.inf
                cube = PUNCHCube(data=corrected_frames[i] * circular_mask, meta=new_meta, wcs=target_frame,
                                 uncertainty=StdDevUncertainty(uncertainty))
                output_cubes.append(cube)

        print("PCA flow done!")
        return output_cubes


def reconstitute(flat_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if flat_image.ndim == 1:
        im = np.zeros(mask.shape)
        im[mask] = flat_image
        return im
    im = np.zeros((len(flat_image), *mask.shape))
    for i in range(len(flat_image)):
        im[i] = reconstitute(flat_image[i], mask)
    return im


def get_pylon_mask(shape: tuple, wcs: WCS) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=float)
    yy -= wcs.wcs.crpix[0] - 1 + 40
    xx -= wcs.wcs.crpix[1] - 1
    ang = np.arctan2(yy, xx) * 180 / np.pi
    mask = (np.abs(ang + 90) > 75) + (np.abs(xx) > 700)
    mask[(np.abs(ang + 90) < 65)] = 0
    # mask = scipy.ndimage.gaussian_filter(mask.astype(np.float32), sigma=15)
    return mask


def _load_one_file(path: str, downsample_factor: int) -> tuple[NormalizedMetadata, WCS, WCS, str, np.ndarray, np.ndarray]:
    cube = load_ndcube_from_fits(path, include_uncertainty=False, include_provenance=False, dtype=np.float32)
    if cube.meta["BADPKTS"].value or cube.meta["DATAP25"].value > 1e-9:
        return None
    l0_path = path.replace("1/XR4", "0/CR4").replace("1_XR4", "0_CR4")
    if not os.path.exists(l0_path):
        l0_path = l0_path.replace('/0/', '/0-old-before-0m/')
    l0 = load_ndcube_from_fits(l0_path)
    data = cube.data
    saturation_mask = l0.data > 1252
    if downsample_factor > 1:
        saturation_mask = saturation_mask.reshape((data.shape[0] // downsample_factor, downsample_factor,
                                                   data.shape[1] // downsample_factor, downsample_factor)).any(axis=(1, 3))
    return cube.meta, cube.wcs, cube.celestial_wcs, path, data, saturation_mask


def load_files(files: list[str], n_workers: int, downsample_factor: int) -> tuple[np.ndarray, list, list, list, list, np.ndarray]:
    metas = []
    wcses = []
    cwcses = []
    loaded_files = []
    x_cube = ShmPickleableNDArray((len(files), 2048, 2048), dtype=np.float32)
    sat_mask_cube = ShmPickleableNDArray((len(files), 2048 // downsample_factor, 2048 // downsample_factor), dtype=bool)
    i = 0
    context = mp.get_context("forkserver")
    with ProcessPoolExecutor(n_workers, mp_context=context) as process_pool:
        for result in process_pool.map(_load_one_file, files, repeat(downsample_factor), chunksize=2):
            # Files that were determined on load to be outliers got rejected
            if result is None:
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


def _fill_one_image(src_data: np.ndarray, dest: np.ndarray, mask_dest: np.ndarray, meta: NormalizedMetadata, wcs: WCS,
                    sat_mask: np.ndarray, nfi_mask: np.ndarray, downsample_factor: int) -> None:
    numba.set_num_threads(2)

    if downsample_factor > 1:
        wcs = wcs[::downsample_factor, ::downsample_factor]
    fill_mask = np.zeros_like(src_data, dtype=bool)
    plot_mask = np.zeros_like(fill_mask, dtype=np.float32)

    body_names = ["moon"]
    bodies = []
    for body in body_names:
        bodies.append(get_body(body, meta.astropy_time))
    bodies = bodies[0] if len(bodies) == 1 else astropy.coordinates.concatenate(bodies)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*failed to converge.*")
        warnings.filterwarnings("ignore", ".*All-NaN slice.*")
        xs, ys = wcs.world_to_pixel(bodies)

        for body, x, y in zip(body_names, np.atleast_1d(xs), np.atleast_1d(ys)):
            if 0 < x < src_data.shape[1] and 0 < y < src_data.shape[0]:
                x, y = int(x), int(y)
                if body != "moon":
                    w = int(round(9 * 2 / downsample_factor))
                else:
                    w = int(round(30 * 2 / downsample_factor))
                    if not np.any(nfi_mask[y - w:y + w + 1, x - w:x + w + 1]):
                        thresh = np.mean(src_data[y - w:y + w + 1, x - w:x + w + 1])
                        while (np.all(src_data[y - w, x - w:x + w + 1] < thresh)
                               and np.all(src_data[y + w, x - w:x + w + 1] < thresh)
                               and np.all(src_data[y - w:y + w + 1, x - w] < thresh)
                               and np.all(src_data[y - w:y + w + 1, x + w] < thresh)):
                            w -= 1
                        w += 1
                fill_mask[y - w:y + w + 1, x - w:x + w + 1] = 1

    sat_idxs = np.nonzero(sat_mask)
    for y, x in zip(*sat_idxs):
        fill_mask[y - 1:y + 2, x - 1:x + 2] = 1
        plot_mask[y - 1:y + 2, x - 1:x + 2] = 1

    for y in np.unique(sat_idxs[0]):
        if np.sum(sat_idxs[0] == y) > 9:
            fill_mask[y] = 1
            plot_mask[y - 1:y + 2] = 1

    filtered_image = nan_percentile_2d(np.where(fill_mask, np.nan, src_data), 50, int(round(5 * 2 / downsample_factor)))
    dest[:] = inpaint_biharmonic(filtered_image, fill_mask)
    mask_dest[:] = plot_mask


def fill_problem_regions(x_cube: np.ndarray, metas: list[NormalizedMetadata], cwcses: list[WCS],
                         sat_mask_cube: np.ndarray, nfi_mask: np.ndarray, downsample_factor: int,
                         process_pool: ProcessPoolExecutor) -> tuple[np.ndarray, np.ndarray]:
    # Copy x_cube
    x_cube_filled = ShmPickleableNDArray.empty_like(x_cube)
    plot_masks = ShmPickleableNDArray.empty_like(x_cube)

    for _ in process_pool.map(_fill_one_image, x_cube, x_cube_filled, plot_masks, metas, cwcses, sat_mask_cube,
                              repeat(nfi_mask), repeat(downsample_factor), chunksize=2):
        # Loop is necessary for any exceptions from workers to be raised
        pass

    return x_cube_filled, plot_masks


def find_outliers_with_PCA(x_cube_filled: np.ndarray, good_mask: np.ndarray, nfi_mask: np.ndarray, n_workers: int) -> np.ndarray:
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
    good_mask = np.ones(len(metas), dtype=bool)
    for k in "DATAP98", "DATAP90", "DATAP50":
        d = np.array([m[k].value for m in metas])
        good_mask[d > np.median(d) + 2 * np.std(d)] = False
    return good_mask


def do_PCA_filtering(x_cube_filled: np.ndarray, good_mask: np.ndarray, nfi_mask: np.ndarray, n_strides: int,
                     n_componenets: int, process_pool: ProcessPoolExecutor, n_workers: int) -> np.ndarray:
    dsls = ShmPickleableNDArray.empty_like(x_cube_filled)
    n_threads = max(1, int(round(n_workers / n_componenets)))
    for _ in process_pool.map(_do_PCA_filtering_one_stride, range(n_strides), repeat(n_strides), repeat(x_cube_filled),
                              repeat(good_mask), repeat(dsls), repeat(nfi_mask), repeat(n_componenets), repeat(n_threads)):
        # Loop is necessary for any exceptions from workers to be raised
        pass
    return dsls


def _do_PCA_filtering_one_stride(stride: int, n_sets: int, x_cube_filled: np.ndarray, good_mask: np.ndarray,
                                 dsls: np.ndarray, nfi_mask: np.ndarray, n_components: int, n_threads: int) -> None:
    phases = (np.arange(len(x_cube_filled)) - stride) % n_sets
    fit_idxs = (phases != 0) * (phases != 1) * (phases != n_sets - 1) * good_mask
    set_to_fit = x_cube_filled[fit_idxs][:, nfi_mask]

    filter_idxs = phases == 0
    set_to_filter = x_cube_filled[filter_idxs][:, nfi_mask]

    pca = PCA(n_components=n_components)
    with limit_threads(n_threads):
        pca.fit(set_to_fit)

        t = pca.transform(set_to_filter)

        comp_smoothing = 5
        for j in range(len(pca.components_)):
            component = reconstitute(pca.components_[j], nfi_mask)
            component = scipy.signal.medfilt2d(component, comp_smoothing)
            pca.components_[j] = component[nfi_mask]

        recon = pca.inverse_transform(t)

    dest_idxs = np.arange(len(x_cube_filled))[filter_idxs]
    dsls[dest_idxs] = reconstitute(recon, nfi_mask)


def subtract_models_from_data(x_cube: np.ndarray, x_cube_ds_filled: np.ndarray, dsl_models: np.ndarray,
                              downsample_factor: int, process_pool: ProcessPoolExecutor) -> tuple[np.ndarray, np.ndarray]:
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


def upsample(image, factor):
    if factor == 1:
        return image
    target_shape = np.array(image.shape) * factor
    binned_indices = (np.arange(0.5 * (factor - 1), target_shape[0], factor),
                      np.arange(0.5 * (factor - 1), target_shape[1], factor))
    upsample_indices = np.stack(np.indices(target_shape), axis=-1)
    interp = RegularGridInterpolator(binned_indices, image, method="linear", bounds_error=False, fill_value=0)
    return interp(upsample_indices)

def downsample(image: np.ndarray, factor: int) -> np.ndarray:
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
    ims1 = scipy.ndimage.percentile_filter(filled_image, 30, size=int(round(11 * 2 / downsample_factor)))
    ims2 = scipy.ndimage.gaussian_filter(filled_image, sigma=15 * 2 / downsample_factor)
    unsharp_image = 0.4 * ims1 + 0.35 * ims2
    unsharp_image = upsample(unsharp_image, downsample_factor)
    unsharp_masked = image - unsharp_image
    destination[:] = scipy.ndimage.percentile_filter(unsharp_masked, 50, size=5)


def inst_frame_filter(filtered_images: np.ndarray, filtered_filled_images: np.ndarray,
                      downsample_factor: int, process_pool: ProcessPoolExecutor) -> np.ndarray:
    post_filtered_images = ShmPickleableNDArray.empty_like(filtered_images)
    for _ in process_pool.map(_inst_frame_filter_one_image, filtered_images, filtered_filled_images, post_filtered_images,
                              repeat(downsample_factor), chunksize=2):
        # Loop is necessary for any exceptions from workers to be raised
        pass

    min_image = nan_percentile(post_filtered_images, 4)
    post_filtered_images -= min_image

    return post_filtered_images


def censor_wcs(wcs):
    """
    Removes observer details from a WCS

    When input images have slightly different viewpoints, Sunpy will say this
    is an invalid coordinate transformation. Here we censor information from the
    WCS to pacify Sunpy.
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
    target_frame = wcses[0].deepcopy()
    target_frame.wcs.pc = np.eye(2)
    target_frame.wcs.crpix = dfiltered_images.shape[2] / 2 + 0.5, dfiltered_images.shape[1] / 2 + 0.5
    target_frame.wcs.crval = 0, 0
    target_frame.cpdis1 = None
    target_frame.cpdis2 = None

    pylon_mask = get_pylon_mask(dfiltered_images[0].shape, wcses[0])

    oriented_images = ShmPickleableNDArray.empty_like(dfiltered_images)
    masks = ShmPickleableNDArray(dfiltered_images.shape, dtype=bool)
    for _ in process_pool.map(_reproject_one_image, dfiltered_images, plot_masks, wcses, repeat(pylon_mask), repeat(target_frame),
                              oriented_images, masks, repeat(downsample_factor)):
        # Loop is necessary for any exceptions from workers to be raised
        pass
    return oriented_images, masks, target_frame


def sinusoid(x: int | float | np.ndarray, dy: np.ndarray, *args) -> np.ndarray:
    if isinstance(x, (int, float)):
        result = dy
    else:
        result = np.full(len(x), dy, dtype=float)
    for f, (A, dphi) in enumerate(batched(args, 2)):
        result += A * np.sin(x * ((f + 1) * np.pi / 180) + dphi)
    return result


def jac(x: np.ndarray, dy: np.ndarray, *args) -> np.ndarray:
    ddy = np.full_like(x, 1)
    ret = [ddy]
    for f, (A, dphi) in enumerate(batched(args, 2)):
        ret.append(np.sin(x * ((f + 1) * np.pi / 180) + dphi))
        ret.append(A * np.cos(x * ((f + 1) * np.pi / 180) + dphi))

    return np.transpose(ret)


def make_correction_map(crota: float, popts: np.ndarray, ivals: np.ndarray) -> np.ndarray:
    map = np.zeros(popts.shape[1:])

    for i in range(len(ivals)):
        for j in range(len(ivals)):
            map[i, j] = sinusoid(crota, *popts[:, i, j])
    return map


def _do_one_sinusoid(args: tuple, crota_vals: np.ndarray, images: np.ndarray, mask: np.ndarray, n_comps: int, whs: int,
                     ivals: np.ndarray) -> list:
    ix, i = args
    rets = []
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*Degrees of freedom.*")
        warnings.filterwarnings("ignore", ".*Mean of empty slice.*")
        for jx, j in enumerate(ivals):
            if np.any(mask[i - whs:i + whs + 1, j - whs:j + whs + 1] == 0):
                continue
            x, y = crota_vals, images[:, i - whs:i + whs + 1, j - whs:j + whs + 1]
            x = np.broadcast_to(x, y.T.shape).T.ravel()
            y = y.ravel()

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

                    popt, _ = scipy.optimize.curve_fit(sinusoid, x, y * 1e12, p0=p0, jac=jac,
                                                       bounds=[lbounds, ubounds],
                                                       loss="cauchy", f_scale=0.5,
                                                       method="trf",
                                                       )
            except RuntimeError:
                popt = np.full(n_comps * 2 + 1, np.nan)
            popt[0] *= 1e-12
            for n in range(1, len(popt), 2):
                popt[n] *= 1e-12
            rets.append((popt, ix, jx))
    return rets


def compute_popts(crota_vals: np.ndarray, images: np.ndarray, mask: np.ndarray, process_pool: ProcessPoolExecutor,
                  n_comps: int, grid_size: int = 110, whs: int = 2) -> tuple[np.ndarray, np.ndarray]:
    ivals = np.round(np.linspace(0, images.shape[1], grid_size)).astype(int)
    popts = np.zeros((2 * n_comps + 1, ivals.size, ivals.size))

    for ret in process_pool.map(_do_one_sinusoid, enumerate(ivals), repeat(crota_vals), repeat(images), repeat(mask),
                                repeat(n_comps), repeat(whs), repeat(ivals)):
        for popt, ix, jx in ret:
            popts[:, ix, jx] = popt
    return popts, ivals


def _desinusoid_one_image(src_image: np.ndarray, crota: float, t: float, ivals: np.ndarray,
                          time_based_popts: np.ndarray, dest: np.ndarray) -> None:
    if t <= time_based_popts[0][0]:
        map = make_correction_map(crota, time_based_popts[0][1], ivals)
    elif t >= time_based_popts[-1][0]:
        map = make_correction_map(crota, time_based_popts[-1][1], ivals)
    else:
        i = 0
        while not time_based_popts[i][0] <= t <= time_based_popts[i + 1][0]:
            i += 1
        t1 = time_based_popts[i][0]
        t2 = time_based_popts[i + 1][0]
        map1 = make_correction_map(crota, time_based_popts[i][1], ivals)
        map2 = make_correction_map(crota, time_based_popts[i + 1][1], ivals)
        map = (t - t1) / (t2 - t1) * (map2 - map1) + map1

    map = scipy.ndimage.median_filter(map, 9)
    interp = RegularGridInterpolator([ivals] * 2, map, bounds_error=False, fill_value=None)
    correction = interp(np.stack(np.mgrid[:src_image.shape[0], :src_image.shape[1]], axis=-1))
    dest[:] = src_image - correction


def do_sinusoid_filtering(oriented_images: np.ndarray, metas: list[NormalizedMetadata], mask: np.ndarray,
                          process_pool: ProcessPoolExecutor) -> np.ndarray:
    crota_vals = np.array([m["CROTA"].value for m in metas])
    time_based_popts = []
    dateobses = np.array([m["DATE-OBS"].value for m in metas], dtype=np.datetime64)

    t0 = dateobses[0]
    dt = np.timedelta64("3", "D")
    while t0 < dateobses[-1]:
        cut = (dateobses > t0) * (dateobses < t0 + dt)
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
