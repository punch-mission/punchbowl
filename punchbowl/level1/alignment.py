import os
import itertools
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

import astrometry
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.wcs import WCS, DistortionLookupTable, NoConvergence, utils
from lmfit import Minimizer, Parameters, minimize
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder
from scipy.ndimage import percentile_filter
from scipy.spatial import KDTree

from punchbowl.data import NormalizedMetadata

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
    catalog: pd.DataFrame, wcs: WCS, image_shape: tuple[int, int], mask: Callable | None = None,
        mode: str = "all", meta: NormalizedMetadata | None = None,
) -> pd.DataFrame:
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

    Returns
    -------
    pd.DataFrame
        pixel coordinates of stars in catalog that are present in the image

    """
    try:
        if meta is None:
            xs, ys = SkyCoord(
                ra=np.array(catalog["RAdeg"]) * u.degree,
                dec=np.array(catalog["DEdeg"]) * u.degree,
                distance=np.array(catalog["Dist_ly"]) * u.lyr,
            ).to_pixel(wcs, mode=mode)
        else:
            sc_location = EarthLocation.from_geodetic(lon=meta["GEOD_LON"].value * u.deg,
                                              lat=meta["GEOD_LAT"].value * u.deg,
                                              height=meta["GEOD_ALT"].value * u.m)
            geoloc, geovel = sc_location.get_gcrs_posvel(meta.astropy_time)
            xs, ys = SkyCoord(
                ra=np.array(catalog["RAdeg"]) * u.degree,
                dec=np.array(catalog["DEdeg"]) * u.degree,
                distance=np.array(catalog["Dist_ly"]) * u.lyr,
                frame="icrs", obsgeoloc=geoloc, obsgeovel=geovel, obstime=meta.astropy_time,
            ).transform_to("gcrs").to_pixel(wcs, mode=mode)
    except NoConvergence as e:
        xs, ys = e.best_solution[:, 0], e.best_solution[:, 1]
    bounds_mask = (xs >= 0) * (xs < image_shape[1]) * (ys >= 0) * (ys < image_shape[0])

    if mask is not None:
        bounds_mask *= mask(xs, ys)

    reduced_catalog = catalog[bounds_mask].copy()
    reduced_catalog["x_pix"] = xs[bounds_mask]
    reduced_catalog["y_pix"] = ys[bounds_mask]
    return reduced_catalog


def find_star_coordinates2(image, return_fluxes=False):
    try:
        sigma_clip = SigmaClip(sigma=3.0)
        bkg_estimator = MedianBackground()
        bkg = Background2D(image, (10, 10), filter_size=(5, 5),
                           sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

        mean, median, std = sigma_clipped_stats(image - bkg.background, sigma=3.0)
        daofind = DAOStarFinder(fwhm=3.0, threshold=5. * std)
        sources = daofind(image - bkg.background)
        sources.sort("flux")

        if return_fluxes:
            return np.stack([sources["xcentroid"], sources["ycentroid"], sources["flux"]], axis=-1)
        return np.stack([sources["xcentroid"], sources["ycentroid"]], axis=-1)

    except Exception as e:
        print(e)
        return np.array([[], []]).T


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
              catalog_tree: KDTree,
              observed_stars: np.ndarray,
              observed_tree: KDTree,
              guess_wcs: WCS) -> float:
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

    errors, _ = get_errors(refined_wcs, catalog_stars, catalog_tree, observed_stars, observed_tree)

    return errors


def get_errors(wcs: WCS, catalog_stars: SkyCoord, catalog_tree: KDTree,
               observed_stars: np.ndarray, observed_tree: KDTree) -> tuple[np.ndarray, np.ndarray]:
    """Compute errors between expected and observed star locations."""
    num_neighbors = 25

    accumulated_error = []
    for closest_observed_star in observed_stars[-num_neighbors:]:
        ra_dec = wcs.pixel_to_world(closest_observed_star[0], closest_observed_star[1])
        catalog_distances, catalog_neighbors = catalog_tree.query([ra_dec.data.lon.to(u.degree).value,
                                                                   ra_dec.data.lat.to(u.degree).value], k=num_neighbors)

        neighbor_ra_dec = catalog_tree.data[catalog_neighbors]

        try:
            neighbor_xs, neighbor_ys = SkyCoord(neighbor_ra_dec[:, 0] * u.degree,
                                                neighbor_ra_dec[:, 1] * u.degree).to_pixel(wcs, mode="all")
        except NoConvergence as e:
            neighbor_xs, neighbor_ys = e.best_solution[:, 0], e.best_solution[:, 1]

        match_distance, match_stars = observed_tree.query(np.stack([neighbor_xs, neighbor_ys], axis=-1), k=1)
        accumulated_error.append(np.sum(match_distance))
    return np.array(accumulated_error), (neighbor_xs, neighbor_ys, closest_observed_star)


def extract_crota_from_wcs(wcs: WCS) -> tuple[float, float]:
    """Extract CROTA from a WCS."""
    delta_ratio = abs(wcs.wcs.cdelt[1]) / abs(wcs.wcs.cdelt[0])
    return (np.arctan2(wcs.wcs.pc[1, 0] / delta_ratio, wcs.wcs.pc[0, 0])) * u.rad


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


def refine_pointing_single_step(
        guess_wcs: WCS, observed_tree: KDTree, catalog_stars: SkyCoord, method: str = "powell",
        ra_tolerance: float = 3, dec_tolerance: float = 3,
        fix_crval: bool = False, fix_crota: bool = False, fix_pv: bool = True) -> WCS:
    """
    Perform a single step of pointing refinement.

    Parameters
    ----------
    guess_wcs : WCS
        the initial guess for the world coordinate system
    observed_tree: KDTree
        coordinates of the observed star positions extracted from the image, as a tree
    catalog_stars : SkyCoord
        the coordinates of known stars to be matched with the observed stars
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
    params.add("crota",
               value=np.random.uniform(0.99, 1.01)*initial_crota.to(u.rad).value,
               min=(initial_crota.to(u.degree) - 3 * u.degree).to(u.rad).value,
               max=(initial_crota.to(u.degree) + 3 * u.degree).to(u.rad).value, vary=not fix_crota)
    params.add("crval1",
               value=np.random.uniform(0.99, 1.01)*guess_wcs.wcs.crval[0],
               min=guess_wcs.wcs.crval[0]-ra_tolerance,
               max=guess_wcs.wcs.crval[0]+ra_tolerance, vary=not fix_crval)
    params.add("crval2",
               value=np.random.uniform(0.99, 1.01)*guess_wcs.wcs.crval[1],
               min=guess_wcs.wcs.crval[1]-dec_tolerance,
               max=guess_wcs.wcs.crval[1]+dec_tolerance, vary=not fix_crval)
    params.add("platescale",
               value=abs(guess_wcs.wcs.cdelt[0]), min=0, max=1, vary=False)
    pv = guess_wcs.wcs.get_pv()[0][-1] if guess_wcs.wcs.get_pv() else 0.0
    params.add("pv", value=pv, min=0.0, max=1.0, vary=not fix_pv)

    observed_stars = observed_tree.data
    catalog_tree = KDTree(np.stack([catalog_stars.data.lon.to(u.degree).value,
                                catalog_stars.data.lat.to(u.degree).value], axis=-1))

    out = minimize(_residual, params, method=method,
                   args=(catalog_stars, catalog_tree, observed_stars, observed_tree, guess_wcs))

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

    return result_wcs, out.residual[0]


def solve_pointing(
        image_data: np.ndarray,
        image_wcs: WCS,
        image_header: NormalizedMetadata,
        distortion: WCS | None = None,
        distance_from_center: float = 300,
        search_scales: list[int] = (14, 15, 16),
        debug: bool = False,
) -> WCS:
    """
    Carefully determine the pointing of an image using the starfield.

    Parameters
    ----------
    image_data : np.ndarray
        a 2D image, preferably with cosmic rays reduced
    image_wcs : WCS
        a guess world coordinate system
    image_header : NormalizedMetadata
        the image's metadata
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
    wcs_arcsec_per_pixel = image_wcs.wcs.cdelt[1] * 3600
    observed = find_star_coordinates2(image_data)

    def mask(observed: np.ndarray) -> np.ndarray:
        distances = np.sqrt(np.square(observed[:, 0] - 1024) + np.square(observed[:, 1] - 1024))
        return distances < distance_from_center

    observed = observed[mask(observed)]

    star_image = image_data - percentile_filter(image_data, 50, 10)
    minimum_brightness = 0
    brightnesses = star_image[observed.astype(int)[:, 1], observed.astype(int)[:, 0]]
    observed_order = np.argsort(brightnesses)[::-1]
    brightness_mask = brightnesses > minimum_brightness
    observed = observed[brightness_mask[observed_order]]
    observed_tree = KDTree(observed)
    # return (), (), observed

    astrometry_net = astrometry_net_initial_solve(observed, image_wcs.deepcopy(),
                                                  search_scales=search_scales,
                                                  lower_arcsec_per_pixel=wcs_arcsec_per_pixel - 10,
                                                  upper_arcsec_per_pixel=wcs_arcsec_per_pixel + 10)
    # astrometry_net = None
    if astrometry_net is None:
        astrometry_net = image_wcs.deepcopy()

    astrometry_net = convert_cd_matrix_to_pc_matrix(astrometry_net)

    image_center = 1024.5, 1024.5  # (image_data.shape[0]//2 + 0.5, image_data.shape[1]//2 + 0.5)
    # image_center = (1021.05, 1023.71)  # TODO don't hard code James's values
    center = astrometry_net.all_pix2world(np.array([image_center]), 0)
    guess_wcs = astrometry_net.deepcopy()
    guess_wcs.wcs.ctype = "RA---AZP", "DEC--AZP"
    guess_wcs.wcs.crval = center[0]
    guess_wcs.wcs.crpix = image_center

    # fp = 34.9811
    # mu = 0.1104
    # sensor_pitch = 15 / 1000
    # platescale = np.arctan(sensor_pitch/fp)
    # platescale = np.rad2deg(platescale)
    # guess_wcs.wcs.cdelt = (-platescale, platescale)

    guess_wcs.wcs.cdelt = image_wcs.wcs.cdelt
    guess_wcs.sip = None
    if distortion is not None:
        guess_wcs.cpdis1 = distortion.cpdis1
        guess_wcs.cpdis2 = distortion.cpdis2
        # if distortion.wcs.get_pv():
        # pv = 0.1104  # TODO don't hard code wfi-1 mu
        # pv = distortion.wcs.get_pv()[0][-1]
        # guess_wcs.wcs.set_pv([(2, 1, pv)])

    catalog = filter_for_visible_stars(load_gaia_catalog(), dimmest_magnitude=9)
    stars_in_image = find_catalog_in_image(catalog, guess_wcs, (2048, 2048), meta=image_header)

    ok_stars = mask(np.stack((stars_in_image["x_pix"], stars_in_image["y_pix"])).T)
    stars_in_image = stars_in_image[ok_stars]
    catalog_stars = SkyCoord(np.array(stars_in_image["RAdeg"]) * u.degree, np.array(stars_in_image["DEdeg"]) * u.degree, frame="icrs")
    # catalog_stars = stars_in_image

    candidate_wcs = [refine_pointing_single_step(guess_wcs, observed_tree, catalog_stars, fix_pv = True)]
    candidate_wcs = [w.result() for w in candidate_wcs]
    errors = [r[1] for r in candidate_wcs]
    candidate_wcs = [r[0] for r in candidate_wcs]
    best = np.argmin(np.abs(errors))
    print(errors[best])
    solved_wcs = candidate_wcs[best]
    if distortion is not None:
        solved_wcs.cpdis1 = distortion.cpdis1
        solved_wcs.cpdis2 = distortion.cpdis2

    if debug:
        return solved_wcs, astrometry_net, observed
    return solved_wcs

# def solve_pointing(
#         image_data: np.ndarray,
#         image_wcs: WCS,
#         image_header: NormalizedMetadata,
#         distortion: WCS | None = None,
#         saturation_limit: float = np.inf,
#         observatory: str = "wfi",
#         n_rounds: int = 175,
#         n_workers: int = 4) -> WCS:
#     """
#     Carefully determine the pointing of an image using the starfield.
#
#     Parameters
#     ----------
#     image_data : np.ndarray
#         a 2D image, preferably with cosmic rays reduced
#     image_wcs : WCS
#         a guess world coordinate system
#     image_header : NormalizedMetadata
#         the image's metadata
#     distortion : WCS | None
#         a distortion WCS to use when fitting
#     saturation_limit : float
#         the maximum star brightness to utilize
#     observatory : str
#         "wfi" or "nfi"
#     n_rounds : int
#         the number of iterations to run for pointing refinement
#     n_workers : int
#         the number of parallel workers to use for pointing refinement
#
#     Returns
#     -------
#     WCS
#         the new world coordinate system
#
#     """
#     logger = get_run_logger()
#
#     wcs_arcsec_per_pixel = image_wcs.wcs.cdelt[1] * 3600
#     if observatory == "wfi":
#         search_scales = (14, 15, 16)
#         max_distance = 700
#         observed = find_star_coordinates(image_data, saturation_limit=saturation_limit, detection_threshold=5.0,
#                                          max_distance_from_center=max_distance)
#
#         def mask(observed: np.ndarray) -> np.ndarray:
#             distances = np.sqrt(np.square(observed[:, 0] - 1024) + np.square(observed[:, 1] - 1024))
#             return distances < max_distance
#     elif observatory == "nfi":
#         search_scales = (11, 12, 13, 14)
#         # We handle max_distance_from_center separately in our mask function, to do it relative to the occulter center
#         observed = find_star_coordinates(image_data, saturation_limit=saturation_limit, detection_threshold=3.0,
#                                          max_distance_from_center=9999)
#
#         def mask(observed: np.ndarray) -> np.ndarray:
#             distances = np.sqrt(np.square(observed[:, 0] - 1013.5) + np.square(observed[:, 1] - 1036.4))
#             distance_mask = distances > 220
#             distance_mask *= distances < 930
#             donut_edge_mask = (distances > 830) * (distances < 870)
#             pylon_mask = (observed[:, 0] > 850) * (observed[:, 0] < 1200) * (observed[:, 1] < 1024)
#             glint_mask = (observed[:, 0] > 475) * (observed[:, 0] < 1550) * (observed[:, 1] < 950) * (
#                     observed[:, 1] > 600)
#             return distance_mask * ~pylon_mask * ~glint_mask * ~donut_edge_mask
#
#
#     else:
#         msg = f"Unknown observatory = {observatory}"
#         raise ValueError(msg)
#     observed = observed[mask(observed)]
#     astrometry_net = astrometry_net_initial_solve(observed, image_wcs.deepcopy(),
#                                                   search_scales=search_scales,
#                                                   lower_arcsec_per_pixel=wcs_arcsec_per_pixel - 10,
#                                                   upper_arcsec_per_pixel=wcs_arcsec_per_pixel + 10)
#     if astrometry_net is None:
#         logger.warning("Astrometry.net initial solution failed. Falling back to spacecraft WCS.")
#         astrometry_net = image_wcs.deepcopy()
#
#     astrometry_net = convert_cd_matrix_to_pc_matrix(astrometry_net)
#
#     image_center = (image_data.shape[0] // 2 + 0.5, image_data.shape[1] // 2 + 0.5)
#     center = astrometry_net.all_pix2world(np.array([image_center]), 0)
#     guess_wcs = astrometry_net.deepcopy()
#     guess_wcs.wcs.ctype = "RA---AZP", "DEC--AZP"
#     guess_wcs.wcs.crval = center[0]
#     guess_wcs.wcs.crpix = image_center
#     guess_wcs.wcs.cdelt = image_wcs.wcs.cdelt
#     guess_wcs.sip = None
#     if distortion is not None:
#         guess_wcs.cpdis1 = distortion.cpdis1
#         guess_wcs.cpdis2 = distortion.cpdis2
#         if distortion.wcs.get_pv():
#             pv = distortion.wcs.get_pv()[0][-1]
#             guess_wcs.wcs.set_pv([(2, 1, pv)])
#
#     catalog = filter_for_visible_stars(load_gaia_catalog(), dimmest_magnitude=7)
#     stars_in_image = find_catalog_in_image(catalog, guess_wcs, (2048, 2048))
#
#     ok_stars = mask(np.stack((stars_in_image["x_pix"], stars_in_image["y_pix"])).T)
#     stars_in_image = stars_in_image[ok_stars]
#
#     catalog_stars = prep_star_coords(stars_in_image, image_header)
#
#     indices = np.arange(len(catalog_stars))
#     rng = np.random.default_rng(seed=1)
#     candidate_wcs = []
#     observed_tree = KDTree(observed)
#     mp_context = multiprocessing.get_context("forkserver")
#     with ProcessPoolExecutor(n_workers, mp_context) as p:
#         for _ in range(n_rounds):
#             sample = catalog_stars[rng.choice(indices, 15, replace=False)]
#             candidate_wcs.append(p.submit(refine_pointing_single_step, guess_wcs, observed_tree, sample, fix_pv=True))
#     candidate_wcs = [w.result() for w in candidate_wcs]
#     errors = [r[1] for r in candidate_wcs]
#     candidate_wcs = [r[0] for r in candidate_wcs]
#     best = np.argmin(np.abs(errors))
#
#     solved_wcs = candidate_wcs[best]
#     if distortion is not None:
#         solved_wcs.cpdis1 = distortion.cpdis1
#         solved_wcs.cpdis2 = distortion.cpdis2
#
#     return solved_wcs

# @punch_task
# def align_task(data_object: NDCube, distortion_path: str | None) -> NDCube:
#     """
#     Determine the pointing of the image and updates the metadata appropriately.
#
#     Parameters
#     ----------
#     data_object : NDCube
#         data object to align
#     distortion_path: str | None
#         path to a distortion model
#
#     Returns
#     -------
#     NDCube
#         a modified version of the input with the WCS more accurately determined
#
#     """
#     celestial_input = calculate_celestial_wcs_from_helio(copy.deepcopy(data_object.wcs),
#                                                          data_object.meta.astropy_time,
#                                                          data_object.data.shape)
#     refining_data = data_object.data.copy()
#     refining_data[np.isinf(refining_data)] = 0
#     refining_data[np.isnan(refining_data)] = 0
#
#     if distortion_path:
#         try:
#             with fits.open(distortion_path) as distortion_hdul:
#                 distortion = WCS(distortion_hdul[0].header, distortion_hdul, key="A")
#         except KeyError:
#             with fits.open(distortion_path) as distortion_hdul:
#                 distortion = WCS(distortion_hdul[0].header, distortion_hdul, key=" ")
#     else:
#         distortion = None
#
#     observatory = "nfi" if data_object.meta["OBSCODE"].value == "4" else "wfi"
#     celestial_output, anet, _ = solve_pointing(cube.data, cube.wcs, cube.meta,
#                                       distortion_models[cube.meta['OBSCODE'].value],
#                                       n_rounds=10, n_workers=8,
#                                       debug=True, star_count=50, distance_from_center=300)
#
#     recovered_wcs = calculate_helio_wcs_from_celestial(celestial_output,
#                                                        data_object.meta.astropy_time,
#                                                        data_object.data.shape)
#
#     if distortion_path:
#         try:
#             with fits.open(distortion_path) as distortion_hdul:
#                 distortion_wcs = WCS(distortion_hdul[0].header, distortion_hdul, key="A")
#         except KeyError:
#             with fits.open(distortion_path) as distortion_hdul:
#                 distortion_wcs = WCS(distortion_hdul[0].header, distortion_hdul, key=" ")
#         recovered_wcs.cpdis1 = distortion_wcs.cpdis1
#         recovered_wcs.cpdis2 = distortion_wcs.cpdis2
#
#     output = NDCube(data=data_object.data,
#                     wcs=recovered_wcs,
#                     uncertainty=data_object.uncertainty,
#                     unit=data_object.unit,
#                     meta=data_object.meta)
#     output.meta.history.add_now("LEVEL1-Align", "alignment done")
#     return output


def solve_patch_lmfit(star_coords, subcatalog, x, y, window_size, initial_guess=None, buffer=0, diagnostic=False):
    try:
        xlow = x
        xhigh = x + window_size
        ylow = y
        yhigh = y + window_size

        xlow_p, xhigh_p = max(xlow - buffer, 0), min(xhigh + buffer, 2048)
        ylow_p, yhigh_p = max(ylow - buffer, 0), min(yhigh + buffer, 2048)

        mask = (star_coords[:, 1] > ylow_p) * (star_coords[:, 1] < yhigh_p) * (star_coords[:, 0] > xlow_p) * (
                    star_coords[:, 0] < xhigh_p)
        observed_coords = star_coords[mask]

        observed_coords[:, 0] -= xlow_p
        observed_coords[:, 1] -= ylow_p

        source_points_2d = observed_coords

        asterism_mask = (subcatalog["x_pix"] > xlow_p) * (subcatalog["y_pix"] > ylow_p) * (
                    subcatalog["x_pix"] < xhigh_p) * (subcatalog["y_pix"] < yhigh_p)
        asterism_catalog = subcatalog[asterism_mask]
        asterism_coords = np.stack([asterism_catalog["x_pix"], asterism_catalog["y_pix"]], axis=-1)

        target_points_2d = asterism_coords.copy()
        target_points_2d[:, 0] -= xlow_p
        target_points_2d[:, 1] -= ylow_p

        params = Parameters()
        if initial_guess is None:
            i_x, i_y = 0, 0
        else:
            i_x, i_y = initial_guess
        params.add("t_x", value=i_x, min=-40, max=40)
        params.add("t_y", value=i_y, min=-40, max=40)

        def translation_residual(params, catalog_stars: np.ndarray, observed_stars: np.ndarray) -> np.ndarray:
            """Compute errors between expected and observed star locations."""
            shift = np.array([params["t_x"], params["t_y"]])

            observed_tree = KDTree(observed_stars + shift)

            errors = np.empty(catalog_stars.shape[0])
            for coord_i, coord in enumerate(catalog_stars):
                dd, ii = observed_tree.query(coord, k=1)
                errors[coord_i] = dd
            return np.nansum(errors)

        # def translation_residual(params, catalog_stars: np.ndarray, observed_stars: np.ndarray) -> np.ndarray:
        #     """Compute errors between expected and observed star locations."""
        #     shift = np.array([params['t_x'], params['t_y']])

        #     catalog_tree = KDTree(catalog_stars+shift)

        #     errors = np.empty(observed_stars.shape[0])
        #     for coord_i, coord in enumerate(observed_stars):
        #         dd, ii = catalog_tree.query(coord, k=1)
        #         errors[coord_i] = dd
        #     return errors

        minner = Minimizer(translation_residual, params, fcn_args=(target_points_2d, source_points_2d))
        result = minner.minimize("least_squares")
        result_shift = np.array([result.params["t_x"].value, result.params["t_y"].value])
        if diagnostic:
            return result_shift, result.chisqr, source_points_2d, target_points_2d
        return result_shift, result.chisqr
    except ValueError:
        return np.array([0, 0]), np.inf


def solve_single_image_distortion(cube, psf_transform=None, catalog=None, initial_distortion=None,
                                  clear_distortion=True):
    if catalog is None:
        catalog = load_gaia_catalog()
        catalog = catalog[catalog["Gmag"] < 10]

    image = cube.data ** 2 / cube.meta["SCALE"].value
    if psf_transform is not None:
        image = psf_transform.apply(image, saturation_threshold=60_000)

    if initial_distortion is not None:
        num_bins = initial_distortion[0].shape[0]
        r = np.linspace(0, 2048, num_bins + 1)
        c = np.linspace(0, 2048, num_bins + 1)
        r = (r[1:] + r[:-1]) / 2
        c = (c[1:] + c[:-1]) / 2

        err_px, err_py = r, c
        cpdis1 = DistortionLookupTable(
            initial_distortion[0].astype(np.float32), (0, 0), (err_px[0], err_py[0]),
            ((err_px[1] - err_px[0]), (err_py[1] - err_py[0])),
        )
        cpdis2 = DistortionLookupTable(
            initial_distortion[1].astype(np.float32), (0, 0), (err_px[0], err_py[0]),
            ((err_px[1] - err_px[0]), (err_py[1] - err_py[0])),
        )
        distortion = WCS(naxis=2)
        distortion.cpdis1 = cpdis1
        distortion.cpdis2 = cpdis2
        distortion.wcs.set_pv([(2, 1, 0.13)])
        solved_wcs = solve_pointing(image.copy(), cube.wcs, cube.meta,
                                    distortion=distortion,
                                    saturation_limit=60_000, set_mu=0.13)
    else:
        solved_wcs = solve_pointing(image.copy(), cube.wcs, cube.meta,
                                    saturation_limit=60_000, set_mu=0.13)
    if clear_distortion:
        solved_wcs.cpdis1 = None
        solved_wcs.cpdis2 = None
    subcatalog = find_catalog_in_image(catalog, solved_wcs, (2048, 2048))

    num_samples = 256
    c = np.linspace(0, 2048, num_samples)
    xx, yy = np.meshgrid(c.copy(), c.copy())
    # xx, yy = xx.flatten(), yy.flatten()

    dmap_x = np.zeros((num_samples, num_samples)) - 1
    dmap_y = np.zeros((num_samples, num_samples)) - 1
    dmap_chi = np.zeros((num_samples, num_samples)) - 1

    star_positions = find_star_coordinates2(image)

    if initial_distortion is None:
        dx = np.zeros((num_samples, num_samples))
        dy = np.zeros_like(dx)
    else:
        dx, dy = initial_distortion

    args = [(star_positions, subcatalog, xx[i, j], yy[i, j], 100, np.array([dx[i, j], dy[i, j]]))
            for i, j in itertools.product(range(num_samples), range(num_samples))]
    with ProcessPoolExecutor(128) as pool:
        out = pool.starmap(solve_patch_lmfit, args)

    for index, (i, j) in enumerate(itertools.product(range(num_samples), range(num_samples))):
        shift, chi = out[index]
        dmap_x[i, j] = shift[0]
        dmap_y[i, j] = shift[1]
        dmap_chi[i, j] = chi

    return dmap_x, dmap_y, dmap_chi
