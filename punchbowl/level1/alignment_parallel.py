# This is the alignment code that's run by the parallel workers. Since we can't fork under prefect, each worker has
# to freshly import the file containing the code it'll run. By moving the code into its own file, we cut the number
# of imports. Each worker's import work drops from ~4.5 s to ~3 s by doing this.


import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS, NoConvergence
from lmfit import Parameters, minimize
from scipy.spatial import KDTree


def refine_pointing_single_step(
        guess_wcs: WCS, observed_tree: KDTree, catalog_stars: SkyCoord, method: str = "least_squares",
        ra_tolerance: float = 10, dec_tolerance: float = 5,
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
    params.add("crota", value=initial_crota.to(u.rad).value,
               min=-np.pi, max=np.pi, vary=not fix_crota)
    params.add("crval1", value=guess_wcs.wcs.crval[0],
               min=guess_wcs.wcs.crval[0] - ra_tolerance,
               max=guess_wcs.wcs.crval[0] + ra_tolerance, vary=not fix_crval)
    params.add("crval2", value=guess_wcs.wcs.crval[1],
               min=guess_wcs.wcs.crval[1] - dec_tolerance,
               max=guess_wcs.wcs.crval[1] + dec_tolerance, vary=not fix_crval)
    params.add("platescale", value=abs(guess_wcs.wcs.cdelt[0]), min=0, max=1, vary=False)
    pv = guess_wcs.wcs.get_pv()[0][-1] if guess_wcs.wcs.get_pv() else 0.0
    params.add("pv", value=pv, min=0.0, max=1.0, vary=not fix_pv)

    with np.errstate(all="ignore"):
        out = minimize(_residual, params, method=method,
                       args=(catalog_stars, observed_tree, guess_wcs),
                       max_nfev=1000, calc_covar=False)
    return (out.params["platescale"].value, out.params["crval1"].value, out.params["crval2"].value,
            out.params["crota"].value, out.params["pv"].value)


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

    errors, _ = get_errors(refined_wcs, catalog_stars, observed_tree)
    errors = errors[errors < max_error]
    return np.nansum(np.abs(errors)) / len(errors)


def get_errors(wcs: WCS, catalog_stars: SkyCoord | tuple[np.ndarray, np.ndarray],
               observed_stars: np.ndarray | KDTree) -> tuple[np.ndarray, np.ndarray]:
    """Compute errors between expected and observed star locations."""
    if isinstance(observed_stars, np.ndarray):
        observed_stars = KDTree(observed_stars)
    if isinstance(catalog_stars, SkyCoord):
        try:
            xs, ys = catalog_stars.to_pixel(wcs, mode="all")
        except NoConvergence as e:
            xs, ys = e.best_solution[:, 0], e.best_solution[:, 1]
    else:
        xs, ys = catalog_stars
    refined_coords = np.stack([xs, ys], axis=-1)

    errors = np.empty(refined_coords.shape[0])
    closest_stars = np.empty(refined_coords.shape)
    for coord_i, coord in enumerate(refined_coords):
        dd, ii = observed_stars.query(coord, k=1)
        errors[coord_i] = dd
        closest_stars[coord_i] = observed_stars.data[ii]

    return errors, closest_stars


def extract_crota_from_wcs(wcs: WCS) -> tuple[float, float]:
    """Extract CROTA from a WCS."""
    delta_ratio = abs(wcs.wcs.cdelt[1]) / abs(wcs.wcs.cdelt[0])
    return (np.arctan2(wcs.wcs.pc[1, 0] / delta_ratio, wcs.wcs.pc[0, 0])) * u.rad
