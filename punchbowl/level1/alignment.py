import os
import copy
import itertools
from collections.abc import Callable

import astrometry
import astropy.units as u
import numpy as np
import pandas as pd
from astropy import modeling
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.wcs import WCS, NoConvergence, utils
from ndcube import NDCube
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder
from scipy.spatial import KDTree

from punchbowl.data import NormalizedMetadata
from punchbowl.data.wcs import calculate_celestial_wcs_from_helio, calculate_helio_wcs_from_celestial
from punchbowl.prefect import punch_task

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
        catalog: pd.DataFrame,
        wcs: WCS, image_shape: tuple[int, int],
        mask: Callable | None = None,
        mode: str = "all",
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
        xs, ys = SkyCoord(
            ra=np.array(catalog["RAdeg"]) * u.degree,
            dec=np.array(catalog["DEdeg"]) * u.degree,
            distance=np.array(catalog["Dist_ly"]) * u.lyr,
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


def prep_star_coords(stars_in_image: pd.DataFrame, image_header: NormalizedMetadata) -> SkyCoord:
    """
    Convert ICRS coordinates to GCRS and put in a SkyCoord that says its ICRS.

    That last bit is for compatibility with the fact that we can't have a "true" GCRS WCS, only RA-DEC that are
    assumed to be ICRS. But as long as it's a consistent set of RA-Dec values, it doesn't matter what frame the
    coordinates think they're in.
    """
    # Convert stellar coordinates to GCRS centered on the spacecraft location
    sc_location = EarthLocation.from_geodetic(lon=image_header["GEOD_LON"].value * u.deg,
                                              lat=image_header["GEOD_LAT"].value * u.deg,
                                              height=image_header["GEOD_ALT"].value * u.m)
    geoloc, geovel = sc_location.get_gcrs_posvel(image_header.astropy_time)
    catalog_stars = SkyCoord(
        np.array(stars_in_image["RAdeg"]) * u.degree,
        np.array(stars_in_image["DEdeg"]) * u.degree,
        np.array(stars_in_image["Dist_ly"]) * u.lyr,
        frame="icrs", obsgeoloc=geoloc, obsgeovel=geovel, obstime=image_header.astropy_time,
    ).transform_to("gcrs")
    return SkyCoord(catalog_stars.ra, catalog_stars.dec, frame="icrs")

def _get_fitted_parameter_from_ensemble_measurements(measurements: np.ndarray,
                                                     histogram_range=0.03,
                                                     guess_stddev=0.05,
                                                     num_bins=50):
    counts, bin_edges = np.histogram(measurements, num_bins,
                                     range=(np.min(measurements) - histogram_range,
                                            np.max(measurements) + histogram_range))
    bin_centers = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
    fitter = modeling.fitting.LevMarLSQFitter()
    model = modeling.models.Gaussian1D(amplitude=np.max(counts), mean=bin_centers[np.argmax(counts)],
                                       stddev=guess_stddev)
    fitted_model = fitter(model, bin_centers, counts)
    return fitted_model.parameters[1]

def find_star_coordinates(image, return_fluxes=False):
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

def solve_center(cube,
                 anet_cutout=400,
                 cutout_width=100,
                 search_scales: list[int] = (14, 15, 16),
                 count=150,
                 num_samples=50_000,
                 error_threshold=1.0 * 0.0225,
                 match_criterion=7.0 * 0.0225,
                 distortion=None):
    anet_start_index = (cube.data.shape[0] - anet_cutout) // 2
    anet_end_index = anet_start_index + anet_cutout
    image = cube.data[anet_start_index:anet_end_index, anet_start_index:anet_end_index]
    anet_cutout_wcs = cube.wcs.deepcopy()
    anet_cutout_wcs.wcs.crpix = anet_cutout / 2 + 0.5, anet_cutout / 2 + 0.5
    wcs_arcsec_per_pixel = anet_cutout_wcs.wcs.cdelt[1] * 3600

    image_positions = find_star_coordinates(image.copy(), return_fluxes=True)

    astrometry_net_wcs = astrometry_net_initial_solve(image_positions,
                                                  anet_cutout_wcs.deepcopy(),
                                                  search_scales=search_scales,
                                                  num_stars=100,
                                                  lower_arcsec_per_pixel=wcs_arcsec_per_pixel - 5,
                                                  upper_arcsec_per_pixel=wcs_arcsec_per_pixel + 5)

    astrometry_net_wcs = convert_cd_matrix_to_pc_matrix(astrometry_net_wcs)

    center = anet_cutout / 2 + 0.5, anet_cutout / 2 + 0.5
    center_radec = astrometry_net_wcs.all_pix2world(np.array([center]), 0)

    guess_wcs = astrometry_net_wcs.deepcopy()
    guess_wcs.wcs.ctype = "RA---AZP", "DEC--AZP"
    guess_wcs.wcs.crval = center_radec[0]
    guess_wcs.wcs.crpix = center
    guess_wcs.wcs.cdelt = cube.wcs.wcs.cdelt
    guess_wcs.sip = None
    guess_wcs.cpdis1 = None
    guess_wcs.cpdis2 = None

    image_positions = np.stack(guess_wcs.pixel_to_world_values(image_positions[:, 0], image_positions[:, 1]), axis=1)
    catalog_found = find_catalog_in_image(load_gaia_catalog(), guess_wcs, image.shape)

    # TODO somewhere we should prep atar coords

    image_positions = image_positions[-int(1.5 * len(catalog_found)):]

    image_position_tree = KDTree(image_positions[:, :2])
    catalog_positions = np.stack([catalog_found["RAdeg"], catalog_found["DEdeg"]], axis=-1)
    catalog_position_tree = KDTree(catalog_positions)

    catalog_count, image_count = count, count

    _, image_ids = image_position_tree.query((cutout_width / 2, cutout_width / 2), k=image_count)
    _, catalog_ids = catalog_position_tree.query((cutout_width / 2, cutout_width / 2), k=catalog_count)

    catalog_triangles = np.array(list(itertools.combinations(catalog_ids, 3)))

    chosen_catalog_triangle_indices = np.random.choice(len(catalog_triangles), size=num_samples, replace=False)

    angles, translations = [], []

    for chosen_catalog_triangle_index in chosen_catalog_triangle_indices:
        ds, matches = image_position_tree.query(catalog_positions[catalog_triangles[chosen_catalog_triangle_index]], k=1)
        if np.all(ds < match_criterion):
            catalog_tri = catalog_positions[catalog_triangles[chosen_catalog_triangle_index]]
            image_tri = image_positions[matches][..., :2]

            image_tri_centroid = np.sum(image_tri, axis=0) / 3
            catalog_tri_centroid = np.sum(catalog_tri, axis=0) / 3

            t = image_tri_centroid - catalog_tri_centroid

            shifted_image_tri = image_tri
            shifted_catalog_tri = catalog_tri + t  # TODO should I really shift?

            ee = image_tri - (catalog_tri + t)
            ee_norm = np.sum(np.linalg.norm(ee, axis=1))  # TODO should include the rotation!

            a1 = (np.atan2(shifted_catalog_tri[0][1] - cutout_width / 2, shifted_catalog_tri[0][0] - cutout_width / 2)
                  - np.atan2(shifted_image_tri[0][1] - cutout_width / 2, shifted_image_tri[0][0] - cutout_width / 2))
            a2 = (np.atan2(shifted_catalog_tri[1][1] - cutout_width / 2, shifted_catalog_tri[1][0] - cutout_width / 2)
                  - np.atan2(shifted_image_tri[1][1] - cutout_width / 2, shifted_image_tri[1][0] - cutout_width / 2))
            a3 = (np.atan2(shifted_catalog_tri[2][1] - cutout_width / 2, shifted_catalog_tri[2][0] - cutout_width / 2)
                  - np.atan2(shifted_image_tri[2][1] - cutout_width / 2, shifted_image_tri[2][0] - cutout_width / 2))
            a = np.rad2deg(np.mean([a1, a2, a3]))

            if ee_norm < error_threshold:
                translations.append(t)
                angles.append(a)

    angles, translations = np.array(angles), np.array(translations)

    dx = _get_fitted_parameter_from_ensemble_measurements(translations[:, 0],
                                                          histogram_range=0.04, guess_stddev=0.05, num_bins=50)
    dy = _get_fitted_parameter_from_ensemble_measurements(translations[:, 1],
                                                          histogram_range=0.04, guess_stddev=0.05, num_bins=50)
    da = _get_fitted_parameter_from_ensemble_measurements(angles,
                                                          histogram_range=0.05, guess_stddev=0.05, num_bins=50)

    new_wcs = guess_wcs.deepcopy()
    cdelt1, cdelt2 = new_wcs.wcs.cdelt
    new_wcs.wcs.crpix = (cube.data.shape[0] + 0.5, cube.data.shape[1] + 0.5)
    new_wcs.wcs.crval -= np.array([dx, dy])
    new_crota = extract_crota_from_wcs(guess_wcs) + da * u.degree
    new_wcs.wcs.pc = np.array(
        [
            [np.cos(new_crota), np.sin(new_crota) * (cdelt1 / cdelt2)],
            [-np.sin(new_crota) * (cdelt2 / cdelt1), np.cos(new_crota)],
        ])

    if distortion is not None:
        new_wcs.cpdis1 = distortion.cpdis1
        new_wcs.cpdis2 = distortion.cpdis2

    return new_wcs

# TODO add code that determines the distortion

@punch_task
def align_task(data_object: NDCube, distortion_path: str | None, max_workers: int = 4) -> NDCube:
    """
    Determine the pointing of the image and updates the metadata appropriately.

    Parameters
    ----------
    data_object : NDCube
        data object to align
    distortion_path: str | None
        path to a distortion model
    max_workers : int
        number of parallel workers to use

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
        try:
            with fits.open(distortion_path) as distortion_hdul:
                distortion = WCS(distortion_hdul[0].header, distortion_hdul, key="A")
        except KeyError:
            with fits.open(distortion_path) as distortion_hdul:
                distortion = WCS(distortion_hdul[0].header, distortion_hdul, key=" ")
    else:
        distortion = None

    observatory = "nfi" if data_object.meta["OBSCODE"].value == "4" else "wfi"
    # TODO : replace with center solver
    celestial_output = solve_pointing(refining_data, celestial_input, data_object.meta, distortion,
                                      saturation_limit=60_000, observatory=observatory, n_workers=max_workers)

    recovered_wcs = calculate_helio_wcs_from_celestial(celestial_output,
                                                       data_object.meta.astropy_time,
                                                       data_object.data.shape)

    if distortion_path:
        try:
            with fits.open(distortion_path) as distortion_hdul:
                distortion_wcs = WCS(distortion_hdul[0].header, distortion_hdul, key="A")
        except KeyError:
            with fits.open(distortion_path) as distortion_hdul:
                distortion_wcs = WCS(distortion_hdul[0].header, distortion_hdul, key=" ")
        recovered_wcs.cpdis1 = distortion_wcs.cpdis1
        recovered_wcs.cpdis2 = distortion_wcs.cpdis2

    output = NDCube(data=data_object.data,
                    wcs=recovered_wcs,
                    uncertainty=data_object.uncertainty,
                    unit=data_object.unit,
                    meta=data_object.meta)
    output.meta.history.add_now("LEVEL1-Align", "alignment done")
    return output
