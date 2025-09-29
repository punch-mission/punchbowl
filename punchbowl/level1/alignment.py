import os
import copy
import warnings
from collections.abc import Callable

import astrometry
import astropy.units as u
import numpy as np
import pandas as pd
import scipy
import sep
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS, DistortionLookupTable, NoConvergence, utils
from lmfit import Parameters, minimize
from ndcube import NDCube
from prefect import get_run_logger
from regularizepsf import ArrayPSFTransform
from scipy.spatial import KDTree
from skimage.transform import resize

from punchbowl.data.wcs import calculate_celestial_wcs_from_helio, calculate_helio_wcs_from_celestial
from punchbowl.prefect import punch_task

_ROOT = os.path.abspath(os.path.dirname(__file__))
HIPPARCOS_URL = "https://cdsarc.cds.unistra.fr/ftp/cats/I/239/hip_main.dat"

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


def load_hipparcos_catalog(catalog_path: str = get_data_path("reduced_hip.csv")) -> pd.DataFrame:
    """
    Load the Hipparcos catalog from the local, reduced version. This version only keeps necessary columns.

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


def load_raw_hipparcos_catalog(catalog_path: str = HIPPARCOS_URL) -> pd.DataFrame:
    """
    Download hipparcos catalog from the website. Not recommended for routine use.

    Parameters
    ----------
    catalog_path : str
        path to the Hipparcos catalog

    Returns
    -------
    pd.DataFrame
        loaded catalog with selected columns

    """
    column_names = (
        "Catalog",
        "HIP",
        "Proxy",
        "RAhms",
        "DEdms",
        "Vmag",
        "VarFlag",
        "r_Vmag",
        "RAdeg",
        "DEdeg",
        "AstroRef",
        "Plx",
        "pmRA",
        "pmDE",
        "e_RAdeg",
        "e_DEdeg",
        "e_Plx",
        "e_pmRA",
        "e_pmDE",
        "DE:RA",
        "Plx:RA",
        "Plx:DE",
        "pmRA:RA",
        "pmRA:DE",
        "pmRA:Plx",
        "pmDE:RA",
        "pmDE:DE",
        "pmDE:Plx",
        "pmDE:pmRA",
        "F1",
        "F2",
        "---",
        "BTmag",
        "e_BTmag",
        "VTmag",
        "e_VTmag",
        "m_BTmag",
        "B-V",
        "e_B-V",
        "r_B-V",
        "V-I",
        "e_V-I",
        "r_V-I",
        "CombMag",
        "Hpmag",
        "e_Hpmag",
        "Hpscat",
        "o_Hpmag",
        "m_Hpmag",
        "Hpmax",
        "HPmin",
        "Period",
        "HvarType",
        "moreVar",
        "morePhoto",
        "CCDM",
        "n_CCDM",
        "Nsys",
        "Ncomp",
        "MultFlag",
        "Source",
        "Qual",
        "m_HIP",
        "theta",
        "rho",
        "e_rho",
        "dHp",
        "e_dHp",
        "Survey",
        "Chart",
        "Notes",
        "HD",
        "BD",
        "CoD",
        "CPD",
        "(V-I)red",
        "SpType",
        "r_SpType",
    )
    df = pd.read_csv(
        catalog_path,
        sep="|",
        names=column_names,
        usecols=["HIP", "Vmag", "RAdeg", "DEdeg", "Plx"],
        na_values=["     ", "       ", "        ", "            "],
    )
    df["distance"] = 1000 / df["Plx"]
    df = df[df["distance"] > 0]
    return df.iloc[np.argsort(df["Vmag"])]


def filter_for_visible_stars(catalog: pd.DataFrame, dimmest_magnitude: float = 6) -> pd.DataFrame:
    """
    Filter to only include stars brighter than a given magnitude.

    Parameters
    ----------
    catalog : pd.DataFrame
        a catalog loaded from `~load_hipparcos_catalog` or `~load_raw_hipparcos_catalog`

    dimmest_magnitude : float
        the dimmest magnitude to keep

    Returns
    -------
    pd.DataFrame`
        a catalog with stars dimmer than the `dimmest_magnitude` removed

    """
    return catalog[catalog["Vmag"] < dimmest_magnitude]


def find_catalog_in_image(
    catalog: pd.DataFrame, wcs: WCS, image_shape: tuple[int, int], mask: Callable | None = None,
        mode: str = "all",
) -> pd.DataFrame:
    """
     Convert the RA/DEC catalog into pixel coordinates using the provided WCS.

    Parameters
    ----------
    catalog : pd.DataFrame
        a catalog loaded from `~thuban.catalog.load_hipparcos_catalog` or `~thuban.catalog.load_raw_hipparcos_catalog`
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

    Returns
    -------
    pd.DataFrame
        pixel coordinates of stars in catalog that are present in the image

    """
    try:
        xs, ys = SkyCoord(
            ra=np.array(catalog["RAdeg"]) * u.degree,
            dec=np.array(catalog["DEdeg"]) * u.degree,
            distance=np.array(catalog["distance"]) * u.parsec,
        ).to_pixel(wcs, mode=mode)
    except NoConvergence as e:
        xs, ys = e.best_solution[:, 0], e.best_solution[:, 1]
    bounds_mask = (xs >= 0) * (xs < image_shape[1]) * (ys >= 0) * (ys < image_shape[0])

    if mask is not None:
        bounds_mask *= mask(xs, ys)

    reduced_catalog = catalog[bounds_mask].copy()
    reduced_catalog["x_pix"] = xs[bounds_mask]
    reduced_catalog["y_pix"] = ys[bounds_mask]
    return reduced_catalog

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

    center = image_data.shape[0]//2, image_data.shape[1]//2
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
                ra_deg=image_wcs.wcs.crval[0],
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


def _residual(params: Parameters,
              catalog_stars: SkyCoord,
              observed_tree: KDTree,
              guess_wcs: WCS,
              max_error: float = 30) -> float:
    """
    Residual used when optimizing the pointing.

    Parameters
    ----------
    params : Parameters
        optimization parameters from lmfit
    catalog_stars : SkyCoord
        image catalog of stars to match against
    observed_tree : KDTree
        a KDTree of the pixel coordinates of the observed stars
    guess_wcs : WCS
        initial guess of the world coordinate system, must overlap with the true WCS
    max_error: float
        stars more distant than this are complete misses, and their error is zeroed out

    Returns
    -------
    np.ndarray
        residual

    """
    refined_wcs = guess_wcs.deepcopy()
    refined_wcs.wcs.cdelt = (-params["platescale"].value, params["platescale"].value)
    refined_wcs.wcs.crval = (params["crval1"].value, params["crval2"].value)
    refined_wcs.wcs.pc = np.array(
        [
            [np.cos(params["crota"]), -np.sin(params["crota"])],
            [np.sin(params["crota"]), np.cos(params["crota"])],
        ],
    )
    refined_wcs.cpdis1 = guess_wcs.cpdis1
    refined_wcs.cpdis2 = guess_wcs.cpdis2

    try:
        xs, ys = catalog_stars.to_pixel(refined_wcs, mode="all")
    except NoConvergence as e:
        xs, ys = e.best_solution[:, 0], e.best_solution[:, 1]
    refined_coords = np.stack([xs, ys], axis=-1)

    out = np.empty(refined_coords.shape[0])
    for coord_i, coord in enumerate(refined_coords):
        dd, _ = observed_tree.query(coord, k=1)
        out[coord_i] = dd

    out[out > max_error] = 0
    return np.nansum(out)

def extract_crota_from_wcs(wcs: WCS) -> tuple[float, float]:
    """Extract CROTA from a WCS."""
    delta_ratio = abs(wcs.wcs.cdelt[1]) / abs(wcs.wcs.cdelt[0])
    return (np.arctan2(wcs.wcs.pc[1, 0]/delta_ratio, wcs.wcs.pc[0, 0])) * u.rad

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


def  refine_pointing_single_step(
    guess_wcs: WCS, observed_coords: np.ndarray, subcatalog: pd.DataFrame, method: str = "least_squares",
                                ra_tolerance: float = 10, dec_tolerance: float = 5,
                                fix_crval: bool = False,
                                fix_crota: bool = False,
                                fix_pv: bool = True) -> WCS:
    """
    Perform a single step of pointing refinement.

    Parameters
    ----------
    guess_wcs : WCS
        the initial guess for the world coordinate system
    observed_coords : np.ndarray
        coordinates of the observed star positions extracted from the image
    subcatalog : pd.DataFrame
        the catalog subset used for this resolution
    method : str
        method used by lmfit for minimization
    ra_tolerance : float
        how many degrees the guess WCS is allowed to be incorrect by in right ascension
    dec_tolerance : float
        how many degrees the guess WCS is allowed to be incorrect by in declination
    fix_crval : bool
        if True the crval is not allowed to vary, otherwise it can be fit
    fix_crota : bool
        if True the crota is not allowed to vary, otherwise it can be fit
    fix_pv : bool
        if True the pv is not allowed to vary, otherwise it can be fit

    Returns
    -------
    WCS
        the new world coordinate system

    """
    # set up the optimization
    params = Parameters()
    initial_crota = extract_crota_from_wcs(guess_wcs)
    params.add("crota", value=initial_crota.to(u.rad).value,
               min=-np.pi, max=np.pi, vary=not fix_crota)
    params.add("crval1", value=guess_wcs.wcs.crval[0],
               min=guess_wcs.wcs.crval[0]-ra_tolerance,
               max=guess_wcs.wcs.crval[0]+ra_tolerance, vary=not fix_crval)
    params.add("crval2", value=guess_wcs.wcs.crval[1],
               min=guess_wcs.wcs.crval[1]-dec_tolerance,
               max=guess_wcs.wcs.crval[1]+dec_tolerance, vary=not fix_crval)
    params.add("platescale", value=abs(guess_wcs.wcs.cdelt[0]), min=0, max=1, vary=False)
    pv = guess_wcs.wcs.get_pv()[0][-1] if guess_wcs.wcs.get_pv() else 0.0
    params.add("pv", value=pv, min=0.0, max=1.0, vary=not fix_pv)

    catalog_stars = SkyCoord(
        np.array(subcatalog["RAdeg"]) * u.degree,
        np.array(subcatalog["DEdeg"]) * u.degree,
        np.array(subcatalog["distance"]) * u.parsec,
    )

    observed_tree = KDTree(observed_coords)

    out = minimize(_residual, params, method=method,
                   args=(catalog_stars, observed_tree, guess_wcs),
                   max_nfev=100, calc_covar=False)
    result_wcs = guess_wcs.deepcopy()
    result_wcs.wcs.cdelt = (-out.params["platescale"].value, out.params["platescale"].value)
    result_wcs.wcs.crval = (out.params["crval1"].value, out.params["crval2"].value)
    result_wcs.wcs.pc = np.array(
        [
            [np.cos(out.params["crota"].value), -np.sin(out.params["crota"].value)],
            [np.sin(out.params["crota"].value), np.cos(out.params["crota"].value)],
        ],
    )
    result_wcs.cpdis1 = guess_wcs.cpdis1
    result_wcs.cpdis2 = guess_wcs.cpdis2
    result_wcs.wcs.set_pv([(2, 1, out.params["pv"].value)])

    return result_wcs

def solve_pointing( # noqa: C901
    image_data: np.ndarray,
    image_wcs: WCS,
    distortion: WCS | None = None,
    saturation_limit: float = np.inf,
    observatory: str = "wfi") -> WCS:
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

    Returns
    -------
    WCS
        the new world coordinate system

    """
    logger = get_run_logger()

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
        def mask(observed:np.ndarray) -> np.ndarray:
            distances = np.sqrt(np.square(observed[:, 0] - 1013.5) + np.square(observed[:, 1] - 1036.4))
            distance_mask = distances > 220
            distance_mask *= distances < 930
            donut_edge_mask = (distances > 830) * (distances < 870)
            pylon_mask = (observed[:, 0] > 850) * (observed[:, 0] < 1200) * (observed[:, 1] < 1024)
            glint_mask = (observed[:, 0] > 475) * (observed[:, 0] < 1550) * (observed[:, 1] < 950) * (
                        observed[:, 1] > 600)
            return distance_mask * ~pylon_mask * ~glint_mask * ~donut_edge_mask

        observed = observed[mask(observed)]
    else:
        msg = f"Unknown observatory = {observatory}"
        raise ValueError(msg)
    astrometry_net = astrometry_net_initial_solve(observed, image_wcs.deepcopy(),
                                                  search_scales=search_scales,
                                                  lower_arcsec_per_pixel=wcs_arcsec_per_pixel - 10,
                                                  upper_arcsec_per_pixel=wcs_arcsec_per_pixel + 10)
    if astrometry_net is None:
        logger.warning("Astrometry.net initial solution failed. Falling back to spacecraft WCS.")
        astrometry_net = image_wcs.deepcopy()

    astrometry_net = convert_cd_matrix_to_pc_matrix(astrometry_net)

    image_center = (image_data.shape[0]//2 + 0.5, image_data.shape[1]//2 + 0.5)
    center = astrometry_net.all_pix2world( np.array([image_center]), 0)
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

    catalog = filter_for_visible_stars(load_hipparcos_catalog(), dimmest_magnitude=8.0)
    stars_in_image = find_catalog_in_image(catalog, guess_wcs, (2048, 2048))

    ok_stars = mask(np.stack((stars_in_image['x_pix'], stars_in_image['y_pix'])).T)
    stars_in_image = stars_in_image[ok_stars]

    candidate_wcs = []
    for _ in range(50):
        sample = stars_in_image.sample(n=15)
        candidate_wcs.append(refine_pointing_single_step(guess_wcs, observed, sample, fix_pv=True))

    ras = [w.wcs.crval[0] for w in candidate_wcs]
    decs = [w.wcs.crval[1] for w in candidate_wcs]
    crotas = [extract_crota_from_wcs(w) for w in candidate_wcs]

    # If we're closer to RA=0 than RA=180, wrap the RAs to avoid trouble if we're straddling the RA=0 line
    if np.abs(ras[0] - 180) > 90:
        ras = np.array(ras)
        ras[ras > 180] -= 360

    solved_wcs = image_wcs.deepcopy()
    solved_wcs.wcs.crval = (np.median(ras) % 360, np.median(decs))
    mean_crota = np.median([c.value for c in crotas])
    cdelt1, cdelt2 = image_wcs.wcs.cdelt
    solved_wcs.wcs.pc = np.array(
        [
            [np.cos(mean_crota), np.sin(mean_crota) * (cdelt1 / cdelt2)],
            [-np.sin(mean_crota) * (cdelt2 / cdelt1), np.cos(mean_crota)],
        ],
    )
    if distortion is not None:
        solved_wcs.cpdis1 = distortion.cpdis1
        solved_wcs.cpdis2 = distortion.cpdis2

    return solved_wcs


def measure_wcs_error(
    image_data: np.ndarray,
    w: WCS,
    dimmest_magnitude: float = 6.0,
    max_error: float = 15.0) -> float:
    """Estimate the error in the WCS based on an image."""
    catalog = filter_for_visible_stars(load_hipparcos_catalog(), dimmest_magnitude=dimmest_magnitude)
    stars_in_image = find_catalog_in_image(catalog, w, image_data.shape)
    try:
        xs, ys = SkyCoord(
            ra=np.array(stars_in_image["RAdeg"]) * u.degree,
            dec=np.array(stars_in_image["DEdeg"]) * u.degree,
            distance=np.array(stars_in_image["distance"]) * u.parsec,
        ).to_pixel(w, mode="all")
    except NoConvergence as e:
        xs, ys = e.best_solution[:, 0], e.best_solution[:, 1]
    refined_coords = np.stack([xs, ys], axis=-1)

    observed_coords = find_star_coordinates(
        image_data,
        detection_threshold = 15.0,
        max_distance_from_center=800,
        saturation_limit=1000)

    out = np.array([np.min(np.linalg.norm(observed_coords - coord, axis=-1)) for coord in refined_coords])
    out[out > max_error] = np.nan
    return np.sqrt(np.nanmean(np.square(out)))

def build_distortion_model(
    l0_paths: list[str],
    dimmest_magnitude: float = 6.5,
    num_bins: int = 60,
    psf_transform: ArrayPSFTransform | None = None) -> WCS:
    """Create a distortion model from a set of PUNCH L0 images."""
    refined_wcses = []
    image_cube = []
    for path in l0_paths:
        with fits.open(path) as hdul:
            image_head = hdul[1].header
            image_data = hdul[1].data.astype(float)
            image_data = image_data ** 2 / image_head["SCALE"]
            if psf_transform is not None:
                saturation_threshold = image_head["DSATVAL"]**2/image_head["SCALE"]*0.9
                image_data = psf_transform.apply(image_data,
                                                 saturation_threshold=saturation_threshold).copy()
            img_shape = image_data.shape
            image_wcs = WCS(hdul[1].header, hdul, key="A")
            mask = image_data != 0

        solved_wcs = solve_pointing(image_data, image_wcs)

        image_cube.append(image_data)
        refined_wcses.append(solved_wcs)

    catalog = filter_for_visible_stars(load_hipparcos_catalog(), dimmest_magnitude=dimmest_magnitude)
    all_distortions = []

    for image_data, new_wcs in zip(image_cube, refined_wcses, strict=False):
        stars_in_image = find_catalog_in_image(catalog, new_wcs, image_data.shape)
        subcoords = np.column_stack([[stars_in_image["RAdeg"], stars_in_image["DEdeg"]]]).T
        refined_coords = new_wcs.all_world2pix(subcoords, 0)

        refined_coords = refined_coords[
            image_data[refined_coords[:, 1].astype(int),
            refined_coords[:, 0].astype(int)] > 10]

        observed_coords = find_star_coordinates(image_data,
            max_distance_from_center=1100,
            detection_threshold=25.0,
            saturation_limit=1000)

        closest_star = np.array([np.argmin(np.linalg.norm(observed_coords - coord, axis=-1))
                                 for coord in refined_coords])
        distance = np.array([np.min(np.linalg.norm(observed_coords - coord, axis=-1))
                                 for coord in refined_coords])
        for i in range(len(closest_star)):
            all_distortions.append({"distance": distance[i],
                                    "ox": observed_coords[closest_star[i]][0],
                                    "oy": observed_coords[closest_star[i]][1],
                                    "nx": refined_coords[i][0],
                                    "ny": refined_coords[i][1]})
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
def align_task(data_object: NDCube, distortion_path: str | None) -> NDCube:
    """
    Determine the pointing of the image and updates the metadata appropriately.

    Parameters
    ----------
    data_object : NDCube
        data object to align
    distortion_path: str | None
        path to a distortion model

    Returns
    -------
    NDCube
        a modified version of the input with the WCS more accurately determined

    """
    celestial_input = calculate_celestial_wcs_from_helio(copy.deepcopy(data_object.wcs),
                                                         data_object.meta.astropy_time,
                                                         data_object.data.shape)
    refining_data = data_object.data.copy()
    refining_data[np.isinf(refining_data)] = 0
    refining_data[np.isnan(refining_data)] = 0

    if distortion_path:
        with fits.open(distortion_path) as distortion_hdul:
            distortion = WCS(distortion_hdul[0].header, distortion_hdul, key="A")
    else:
        distortion = None

    observatory = "nfi" if data_object.meta["OBSCODE"].value == "4" else "wfi"
    celestial_output = solve_pointing(refining_data, celestial_input, distortion,
                                      saturation_limit=60_000, observatory=observatory)

    recovered_wcs, _ = calculate_helio_wcs_from_celestial(celestial_output,
                                                       data_object.meta.astropy_time,
                                                       data_object.data.shape)

    if distortion_path:
        with fits.open(distortion_path) as distortion_hdul:
            distortion_wcs = WCS(distortion_hdul[0].header, distortion_hdul, key="A")
        recovered_wcs.cpdis1 = distortion_wcs.cpdis1
        recovered_wcs.cpdis2 = distortion_wcs.cpdis2

    output = NDCube(data=data_object.data,
                    wcs=recovered_wcs,
                    uncertainty=data_object.uncertainty,
                    unit=data_object.unit,
                    meta=data_object.meta)
    output.meta.history.add_now("LEVEL1-Align", "alignment done")
    return output
