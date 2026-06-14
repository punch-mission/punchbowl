import os
import logging
import warnings
import multiprocessing
from pathlib import Path
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

import astrometry
import astropy.units as u
import numpy as np
import pandas as pd
import scipy
import sep
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS, DistortionLookupTable, NoConvergence, utils, FITSFixedWarning
from regularizepsf import ArrayPSFTransform
from scipy.spatial import KDTree
from skimage.transform import resize

from punchbowl.data import NormalizedMetadata
from punchbowl.data.punchcube import PUNCHCube
from punchbowl.data.wcs import calculate_helio_wcs_from_celestial
from punchbowl.level1.alignment_parallel import get_errors, refine_pointing_single_step
from punchbowl.prefect import get_logger, punch_task

_ROOT = os.path.abspath(os.path.dirname(__file__))


def download_gaia_data(out_path: str, dimmest_mag: float = 9) -> None:
    """Download and pre-process Gaia data."""
    from astroquery.gaia import Gaia  # noqa: PLC0415
    query = f"""SELECT source_id, ra, dec, phot_g_mean_mag, parallax from gaiadr3.gaia_source
                WHERE phot_g_mean_mag < {dimmest_mag}
                 AND dec > -70
                 AND dec < 70
            """  # noqa: S608
    job = Gaia.launch_job_async(query)
    results = job.get_results()

    # Remove the few records with no parallax
    results = results[~results["parallax"].mask]

    results["Dist_ly"] = np.round(3.26 / (results["parallax"] / 1000), 2)
    results.remove_column("parallax")
    results = results[results["Dist_ly"] > 2]

    results.rename_column("ra", "RAdeg")
    results.rename_column("dec", "DEdeg")
    results.rename_column("phot_g_mean_mag", "Gmag")

    # Removing digits we don't need to cut the file size
    results["Gmag"] = np.round(results["Gmag"], 1)
    results["RAdeg"] = np.round(results["RAdeg"], 7)
    results["DEdeg"] = np.round(results["DEdeg"], 7)
    results.to_pandas(index="source_id").to_csv(out_path)


def filter_distortion_table(data: np.ndarray, blur_sigma: float = 4, med_filter_size: float = 3) -> np.ndarray:
    """
    Filter a copy of the distortion lookup table.

    Any rows/columns at the edges that are all NaNs will be removed and
    replaced with a copy of the closest non-removed edge at the end of
    processing.

    Any NaN values that don't form a complete edge row/column will be replaced
    with the median of all surrounding non-NaN pixels.

    Then median filtering is performed across the whole map to remove outliers,
    and Gaussian filtering is applied to accept only slowly-varying
    distortions.

    Parameters
    ----------
    data
        The distortion map to be filtered
    blur_sigma : float
        The number of pixels constituting one standard deviation of the
        Gaussian kernel. Set to 0 to disable Gaussian blurring.
    med_filter_size : int
        The size of the local neighborhood to consider for median filtering.
        Set to 0 to disable median filtering.

    Notes
    -----
    Modified from https://github.com/svank/wispr_analysis/blob/main/wispr_analysis/image_alignment.py

    """
    data = data.copy()

    # Trim empty (all-nan) rows and columns
    trimmed = []
    i = 0
    while np.all(np.isnan(data[0])):
        i += 1
        data = data[1:]
    trimmed.append(i)

    i = 0
    while np.all(np.isnan(data[-1])):
        i += 1
        data = data[:-1]
    trimmed.append(i)

    i = 0
    while np.all(np.isnan(data[:, 0])):
        i += 1
        data = data[:, 1:]
    trimmed.append(i)

    i = 0
    while np.all(np.isnan(data[:, -1])):
        i += 1
        data = data[:, :-1]
    trimmed.append(i)

    # Replace interior nan values with the median of the surrounding values.
    # We're filling in from neighboring pixels, so if there are any nan pixels
    # fully surrounded by nan pixels, we need to iterate a few times.
    while np.any(np.isnan(data)):
        nans = np.nonzero(np.isnan(data))
        replacements = np.zeros_like(data)
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", message="All-NaN slice")
            for r, c in zip(*nans, strict=False):
                r1, r2 = r - 1, r + 2
                c1, c2 = c - 1, c + 2
                r1, r2 = max(r1, 0), min(r2, data.shape[0])
                c1, c2 = max(c1, 0), min(c2, data.shape[1])

                replacements[r, c] = np.nanmedian(data[r1:r2, c1:c2])
        data[nans] = replacements[nans]

    # Median-filter the whole image
    if med_filter_size:
        data = scipy.ndimage.median_filter(data, size=med_filter_size, mode="reflect")

    # Gaussian-blur the whole image
    if blur_sigma > 0:
        data = scipy.ndimage.gaussian_filter(data, sigma=blur_sigma)

    # Replicate the edge rows/columns to replace those we trimmed earlier
    return np.pad(data, [trimmed[0:2], trimmed[2:]], mode="edge")


def get_data_path(path: str) -> str:
    """Get the path to the local data directory."""
    return os.path.join(_ROOT, "data", path)


def load_gaia_catalog(catalog_path: str = get_data_path("gaia_catalog.csv")) -> pd.DataFrame:
    """
    Load the Gaia catalog from the local stash.

    Parameters
    ----------
    catalog_path : str
        path to the catalog, defaults to a provided version

    Returns
    -------
    pd.DataFrame
        loaded catalog with selected columns

    """
    return pd.read_csv(catalog_path)


def filter_for_visible_stars(catalog: pd.DataFrame, dimmest_magnitude: float = 6) -> pd.DataFrame:
    """
    Filter to only include stars brighter than a given magnitude.

    Parameters
    ----------
    catalog : pd.DataFrame
        a catalog loaded from `~load_gaia_catalog` or `~load_raw_gaia_catalog`

    dimmest_magnitude : float
        the dimmest magnitude to keep

    Returns
    -------
    pd.DataFrame`
        a catalog with stars dimmer than the `dimmest_magnitude` removed

    """
    return catalog[catalog["Gmag"] < dimmest_magnitude]


def find_catalog_in_image(
        catalog: SkyCoord, wcs: WCS, image_shape: tuple[int, int], mask: Callable | None = None,
        mode: str = "all", dataframe: pd.DataFrame | None = None,
) -> tuple[SkyCoord, np.ndarray]:
    """
     Convert the RA/DEC catalog into pixel coordinates using the provided WCS.

    Parameters
    ----------
    catalog : pd.DataFrame
        a catalog loaded from `~load_gaia_catalog`
    wcs : WCS
        the world coordinate system of a given image
    image_shape: (int, int)
        the shape of the image array associated with the WCS, used to only consider stars with coordinates in image
    mask: Callable
        a function that indicates whether a given coordinate is included
    mode : str
        either "all" or "wcs",
        see
        <https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html#astropy.coordinates.SkyCoord.to_pixel>
    dataframe : pd.DataFrame
        optional data frame to filter and return as well

    Returns
    -------
    pd.DataFrame
        pixel coordinates of stars in catalog that are present in the image

    """
    try:
        xs, ys = catalog.to_pixel(wcs, mode=mode)
    except NoConvergence as e:
        xs, ys = e.best_solution[:, 0], e.best_solution[:, 1]
    bounds_mask = (xs >= 0) * (xs < image_shape[1]) * (ys >= 0) * (ys < image_shape[0])

    if mask is not None:
        bounds_mask *= mask(xs, ys)

    reduced_catalog = catalog[bounds_mask]
    coords = np.stack((xs[bounds_mask], ys[bounds_mask]), axis=1)
    ret = reduced_catalog, coords
    if dataframe is not None:
        dataframe = dataframe[bounds_mask]
        ret = (*ret, dataframe)
    return ret


def find_star_coordinates(image_data: np.ndarray,
                          saturation_limit: float = np.inf,
                          max_distance_from_center: float = 700,
                          background_size: int = 16,
                          detection_threshold: float = 5.0) -> np.ndarray:
    """
    Extract the coordinates of observed stars in an image using sep.

    Parameters
    ----------
    image_data : np.ndarray
        an array of an image
    saturation_limit : float
        stars brighter than this are ignored
    max_distance_from_center: float
        only returns stars at most this distance from the center of the image
    background_size: int
        pixel size used by sep when building background model
    detection_threshold : float
        number of sigma brighter than noise level a star must be for detection

    Returns
    -------
    np.ndarray
        pixel coordinates of stars that are present in the image

    """
    image_copy = image_data.copy()
    image_copy[image_copy > saturation_limit] = 0
    if background_size > 0:
        background = sep.Background(image_data, bw=background_size, bh=background_size)
        image_sub = image_data - background
        objects = sep.extract(image_sub, detection_threshold, err=background.globalrms)
    else:
        image_sub = image_data
        objects = sep.extract(image_sub, detection_threshold)
    objects = pd.DataFrame(objects).sort_values("flux")
    observed_coords = np.stack([objects["x"], objects["y"]], axis=-1)

    center = image_data.shape[0] // 2, image_data.shape[1] // 2
    distance = np.sqrt(np.square(observed_coords[:, 0] - center[0]) + np.square(observed_coords[:, 1] - center[1]))
    return observed_coords[distance < max_distance_from_center, :]


def astrometry_net_initial_solve(observed_coords: np.ndarray,
                                 image_wcs: WCS,
                                 search_scales: tuple[int] = (14, 15, 16),
                                 num_stars: int = 150,
                                 lower_arcsec_per_pixel: float = 80.0,
                                 upper_arcsec_per_pixel: float = 100.0) -> WCS | None:
    """
    Solve for the WCS of an image using Astrometry.net.

    Parameters
    ----------
    observed_coords : np.ndarray
        pixel coordinates of stars in image, returned by `find_star_coordinates`
    image_wcs : WCS
        best guess WCS
    search_scales: tuple[int]
        scales to use for search, see https://github.com/neuromorphicsystems/astrometry?tab=readme-ov-file#choosing-series
    num_stars: int
        number of stars in the observed_coords to use for search
    lower_arcsec_per_pixel: float
        lower guess on the platescale
    upper_arcsec_per_pixel: float
        upper guess on the platescale

    Returns
    -------
    WCS | None
        the best WCS if search successful, otherwise None

    """
    # Astrometry sends INFO messages to this logger, which we don't really want. There doesn't seem to be a context
    # manager for log levels, so we grab the current log level and restore it at the end.
    logger = logging.getLogger("root")
    original_log_level = logger.level
    try:
        logger.setLevel(logging.WARNING)
        with astrometry.Solver(
                astrometry.series_4100.index_files(
                    cache_directory="astrometry_cache",
                    scales=search_scales,
                ),
        ) as solver:
            solution = solver.solve(
                stars=observed_coords[-num_stars:],
                size_hint=astrometry.SizeHint(
                    lower_arcsec_per_pixel=lower_arcsec_per_pixel,
                    upper_arcsec_per_pixel=upper_arcsec_per_pixel,
                ),
                position_hint=astrometry.PositionHint(
                    ra_deg=image_wcs.wcs.crval[0] % 360,
                    dec_deg=image_wcs.wcs.crval[1],
                    radius_deg=15,
                ),
                solution_parameters=astrometry.SolutionParameters(
                    sip_order=0,
                    tune_up_logodds_threshold=None,
                    parity=astrometry.Parity.NORMAL,
                ),
            )
        if solution.has_match():
            return solution.best_match().astropy_wcs()
        return None
    finally:
        logger.setLevel(original_log_level)


def convert_cd_matrix_to_pc_matrix(wcs: WCS) -> WCS:
    """Convert a WCS with a CD matrix to one with a PC matrix."""
    if not hasattr(wcs.wcs, "cd"):
        return wcs
    cdelt1, cdelt2 = utils.proj_plane_pixel_scales(wcs)
    crota = np.arctan2(abs(cdelt1) * wcs.wcs.cd[0, 1], abs(cdelt2) * wcs.wcs.cd[0, 0])

    new_wcs = WCS(naxis=2)
    new_wcs.wcs.ctype = wcs.wcs.ctype
    new_wcs.wcs.crval = wcs.wcs.crval
    new_wcs.wcs.crpix = wcs.wcs.crpix
    new_wcs.wcs.pc = np.array(
        [
            [-np.cos(crota), -np.sin(crota) * (cdelt1 / cdelt2)],
            [np.sin(crota) * (cdelt2 / cdelt1), -np.cos(crota)],
        ])
    new_wcs.wcs.cdelt = (-cdelt1, cdelt2)
    new_wcs.wcs.cunit = "deg", "deg"
    return new_wcs


def solve_pointing( # noqa: C901
        image_data: np.ndarray,
        image_wcs: WCS,
        distortion: WCS | None = None,
        saturation_limit: float = np.inf,
        observatory: str = "wfi",
        n_rounds: int = 50,
        n_workers: int = 4) -> WCS:
    """
    Carefully determine the pointing of an image using the starfield.

    Parameters
    ----------
    image_data : np.ndarray
        a 2D image, preferably with cosmic rays reduced
    image_wcs : WCS
        a guess world coordinate system
    distortion : WCS | None
        a distortion WCS to use when fitting
    saturation_limit : float
        the maximum star brightness to utilize
    observatory : str
        "wfi" or "nfi"
    n_rounds : int
        the number of iterations to run for pointing refinement
    n_workers : int
        the number of parallel workers to use for pointing refinement

    Returns
    -------
    WCS
        the new world coordinate system

    """
    logger = get_logger()

    wcs_arcsec_per_pixel = image_wcs.wcs.cdelt[1] * 3600
    if observatory == "wfi":
        search_scales = (14, 15, 16)
        max_distance = 700
        observed = find_star_coordinates(image_data, saturation_limit=saturation_limit, detection_threshold=5.0,
                                         max_distance_from_center=max_distance)

        def mask(observed: np.ndarray) -> np.ndarray:
            distances = np.sqrt(np.square(observed[:, 0] - 1024) + np.square(observed[:, 1] - 1024))
            return distances < max_distance
    elif observatory == "nfi":
        search_scales = (11, 12, 13, 14)
        # We handle max_distance_from_center separately in our mask function, to do it relative to the occulter center
        observed = find_star_coordinates(image_data, saturation_limit=saturation_limit, detection_threshold=3.0,
                                         max_distance_from_center=9999)

        def mask(observed: np.ndarray) -> np.ndarray:
            distances = np.sqrt(np.square(observed[:, 0] - 1013.5) + np.square(observed[:, 1] - 1036.4))
            distance_mask = distances > 220
            distance_mask *= distances < 930
            donut_edge_mask = (distances > 830) * (distances < 870)
            pylon_mask = (observed[:, 0] > 850) * (observed[:, 0] < 1200) * (observed[:, 1] < 1024)
            glint_mask = (observed[:, 0] > 475) * (observed[:, 0] < 1550) * (observed[:, 1] < 950) * (
                    observed[:, 1] > 600)
            return distance_mask * ~pylon_mask * ~glint_mask * ~donut_edge_mask


    else:
        msg = f"Unknown observatory = {observatory}"
        raise ValueError(msg)
    observed = observed[mask(observed)]
    astrometry_net = astrometry_net_initial_solve(observed, image_wcs.deepcopy(),
                                                  search_scales=search_scales,
                                                  lower_arcsec_per_pixel=wcs_arcsec_per_pixel - 10,
                                                  upper_arcsec_per_pixel=wcs_arcsec_per_pixel + 10)
    if astrometry_net is None:
        logger.warning("Astrometry.net initial solution failed. Falling back to spacecraft WCS.")
        astrometry_net = image_wcs.deepcopy()

    astrometry_net = convert_cd_matrix_to_pc_matrix(astrometry_net)

    image_center = (image_data.shape[0] // 2 + 0.5, image_data.shape[1] // 2 + 0.5)
    center = astrometry_net.all_pix2world(np.array([image_center]), 0)
    guess_wcs = astrometry_net.deepcopy()
    guess_wcs.wcs.ctype = "RA---AZP", "DEC--AZP"
    guess_wcs.wcs.crval = center[0]
    guess_wcs.wcs.crpix = image_center
    guess_wcs.wcs.cdelt = image_wcs.wcs.cdelt
    guess_wcs.sip = None
    if distortion is not None:
        guess_wcs.cpdis1 = distortion.cpdis1
        guess_wcs.cpdis2 = distortion.cpdis2
        if distortion.wcs.get_pv():
            pv = distortion.wcs.get_pv()[0][-1]
            guess_wcs.wcs.set_pv([(2, 1, pv)])

    catalog_stars = filter_for_visible_stars(load_gaia_catalog(), dimmest_magnitude=7)
    catalog_stars = prep_star_coords(catalog_stars, image_wcs)
    catalog_stars, pix_coords = find_catalog_in_image(catalog_stars, guess_wcs, (2048, 2048))

    ok_stars = mask(np.stack((pix_coords[:, 0], pix_coords[:, 1])).T)
    catalog_stars = catalog_stars[ok_stars]

    indices = np.arange(len(catalog_stars))
    rng = np.random.default_rng(seed=1)
    results = []
    observed_tree = KDTree(observed)
    mp_context = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(n_workers, mp_context) as p:
        for _ in range(n_rounds):
            sample = catalog_stars[rng.choice(indices, 15, replace=False)]
            results.append(p.submit(refine_pointing_single_step, guess_wcs, observed_tree, sample, fix_pv=True))
    results = [w.result() for w in results]

    platescales, crval1s, crval2s, crotas, pvs = zip(*results, strict=True)
    solved_wcs = guess_wcs
    cdelt = np.median(platescales)
    solved_wcs.wcs.cdelt = -cdelt, cdelt

    crval1s = np.array(crval1s)
    if np.any(crval1s < 5) and np.any(crval1s > 355):
        # We're straddling the wrap point, at 360 -> 0 deg
        # Shift the high values down to -180-or-so
        crval1s[crval1s > 180] -= 360
        crval1 = np.median(crval1s)
        crval1 %= 360
    else:
        crval1 = np.median(crval1s) % 360

    solved_wcs.wcs.crval = crval1, np.median(crval2s)
    crotas = np.array(crotas)
    if np.any(crotas < -170 * np.pi / 180) and np.any(crotas > 170 * np.pi / 180):
        # We're straddling the wrap point, at 180 -> -180 deg
        # Shift the negative values up to 180 + change
        crotas %= 2 * np.pi
    crota = np.median(crotas)
    solved_wcs.wcs.pc = np.array(
        [
            [np.cos(crota), -np.sin(crota)],
            [np.sin(crota), np.cos(crota)],
        ],
    )
    solved_wcs.wcs.set_pv([(2, 1, np.median(pvs))])

    return solved_wcs


def prep_star_coords(stars_in_image: pd.DataFrame, image_wcs: WCS) -> SkyCoord:
    """Convert ICRS coordinates to GCRS."""
    # Convert stellar coordinates to GCRS centered on the spacecraft location
    return SkyCoord(
        np.array(stars_in_image["RAdeg"]) * u.degree,
        np.array(stars_in_image["DEdeg"]) * u.degree,
        np.array(stars_in_image["Dist_ly"]) * u.lyr,
        frame="icrs",
    ).transform_to(image_wcs.pixel_to_world(0,0).frame)


def measure_wcs_error(
        image_data: np.ndarray,
        wcs: WCS,
        dimmest_magnitude: float = 6.0,
        max_error: float = 15.0, debug: bool = True) -> float:
    """Estimate the error in the WCS based on an image."""
    catalog_stars = filter_for_visible_stars(load_gaia_catalog(), dimmest_magnitude=dimmest_magnitude)
    catalog_stars = prep_star_coords(catalog_stars, wcs)
    catalog_stars, _ = find_catalog_in_image(catalog_stars, wcs, image_data.shape)

    observed_coords = find_star_coordinates(
        image_data,
        detection_threshold=15.0,
        max_distance_from_center=800,
        saturation_limit=1000)

    errors, _ = get_errors(wcs, catalog_stars, observed_coords)

    errors = errors[errors <= max_error]
    if debug:
        return np.sqrt(np.mean(np.square(errors))), errors
    return np.sqrt(np.mean(np.square(errors)))


def build_distortion_model(
        l0_paths: list[str],
        dimmest_magnitude: float = 6.5,
        num_bins: int = 60,
        psf_transform: ArrayPSFTransform | None = None) -> WCS:
    """Create a distortion model from a set of PUNCH L0 images."""
    refined_wcses = []
    image_cube = []
    image_metas = []
    for path in l0_paths:
        with fits.open(path) as hdul:
            image_head = hdul[1].header
            image_data = hdul[1].data.astype(float)
            image_data = image_data ** 2 / image_head["SCALE"]
            if psf_transform is not None:
                saturation_threshold = image_head["DSATVAL"] ** 2 / image_head["SCALE"] * 0.9
                image_data = psf_transform.apply(image_data,
                                                 saturation_threshold=saturation_threshold).copy()
            img_shape = image_data.shape
            image_wcs = WCS(hdul[1].header, hdul, key="A")
            mask = image_data != 0

        meta = NormalizedMetadata.from_fits_header(image_head)
        solved_wcs = solve_pointing(image_data, image_wcs, meta)

        image_cube.append(image_data)
        refined_wcses.append(solved_wcs)
        image_metas.append(meta)

    catalog = filter_for_visible_stars(load_gaia_catalog(), dimmest_magnitude=dimmest_magnitude)
    all_distortions = []

    for image_data, new_wcs in zip(image_cube, refined_wcses, strict=False):
        catalog_stars = prep_star_coords(catalog, new_wcs)
        _, expected_coords = find_catalog_in_image(catalog_stars, new_wcs, image_data.shape)

        observed_coords = find_star_coordinates(image_data,
                                                max_distance_from_center=1200,
                                                detection_threshold=25.0,
                                                saturation_limit=1000)

        distances, matched_stars = get_errors(new_wcs, expected_coords, observed_coords)

        for i in range(len(expected_coords)):
            all_distortions.append({"distance": distances[i],
                                    "ox": matched_stars[i][0],
                                    "oy": matched_stars[i][1],
                                    "nx": expected_coords[i][0],
                                    "ny": expected_coords[i][1]})
    df = pd.DataFrame(all_distortions)

    xbins, r, c, _ = scipy.stats.binned_statistic_2d(
        df["oy"],
        df["ox"],
        df["ox"] - df["nx"],
        "median",
        (num_bins, num_bins),
        expand_binnumbers=True,
        range=((0, img_shape[1]), (0, img_shape[0])),
    )

    ybins, _, _, _ = scipy.stats.binned_statistic_2d(
        df["oy"],
        df["ox"],
        df["oy"] - df["ny"],
        "median",
        (num_bins, num_bins),
        expand_binnumbers=True,
        range=((0, img_shape[1]), (0, img_shape[0])),
    )

    mask = resize(mask, (num_bins, num_bins))

    xbins *= mask
    ybins *= mask

    xbins = filter_distortion_table(xbins, 1.1, 1) * mask
    ybins = filter_distortion_table(ybins, 1.1, 1) * mask

    r = np.linspace(0, 2048, num_bins + 1)
    c = np.linspace(0, 2048, num_bins + 1)
    r = (r[1:] + r[:-1]) / 2
    c = (c[1:] + c[:-1]) / 2

    err_px, err_py = r, c
    cpdis1 = DistortionLookupTable(
        -xbins.astype(np.float32), (0, 0), (err_px[0], err_py[0]),
        ((err_px[1] - err_px[0]), (err_py[1] - err_py[0])),
    )
    cpdis2 = DistortionLookupTable(
        -ybins.astype(np.float32), (0, 0), (err_px[0], err_py[0]),
        ((err_px[1] - err_px[0]), (err_py[1] - err_py[0])),
    )

    out_wcs = solved_wcs.copy()
    out_wcs.cpdis1 = cpdis1
    out_wcs.cpdis2 = cpdis2

    return out_wcs


@punch_task
def align_task(data_object: PUNCHCube, distortion_path: str | WCS | None, max_workers: int = 4,
               n_rounds: int = 50) -> PUNCHCube:
    """
    Determine the pointing of the image and updates the metadata appropriately.

    Parameters
    ----------
    data_object : PUNCHCube
        data object to align
    distortion_path: str | None
        path to a distortion model
    max_workers : int
        number of parallel workers to use
    n_rounds : int
        number of iterations for alignment

    Returns
    -------
    PUNCHCube
        a modified version of the input with the WCS more accurately determined

    """
    celestial_input = data_object.celestial_wcs
    refining_data = data_object.data.copy()
    refining_data[np.isinf(refining_data)] = 0
    refining_data[np.isnan(refining_data)] = 0

    if isinstance(distortion_path, (str, Path)):
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", message=".*The WCS transformation has more axes.*",
                                    category=FITSFixedWarning)
            try:
                with fits.open(distortion_path) as distortion_hdul:
                    distortion = WCS(distortion_hdul[0].header, distortion_hdul, key="A")
            except KeyError:
                with fits.open(distortion_path) as distortion_hdul:
                    distortion = WCS(distortion_hdul[0].header, distortion_hdul, key=" ")
    else:
        distortion = distortion_path

    observatory = "nfi" if data_object.meta["OBSCODE"].value == "4" else "wfi"
    celestial_output = solve_pointing(refining_data, celestial_input, distortion,
                                      saturation_limit=60_000, observatory=observatory, n_workers=max_workers,
                                      n_rounds=n_rounds)

    recovered_wcs = calculate_helio_wcs_from_celestial(celestial_output,
                                                       data_object.meta.astropy_time,
                                                       data_object.data.shape)

    output = data_object.replace(wcs=recovered_wcs, celestial_wcs=celestial_output)
    output.meta.history.add_now("LEVEL1-Align", f"alignment done with {n_rounds} iterations")
    return output
