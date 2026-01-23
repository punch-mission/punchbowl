import pathlib
import warnings
import multiprocessing
from datetime import UTC, datetime
from functools import cached_property
from collections.abc import Generator

import numpy as np
import scipy
from astropy.nddata import StdDevUncertainty
from dateutil.parser import parse as parse_datetime
from lmfit import Parameters, minimize
from lmfit.minimizer import MinimizerResult
from ndcube import NDCube
from prefect import get_run_logger

from punchbowl.data import NormalizedMetadata, load_ndcube_from_fits
from punchbowl.data.punch_io import load_many_cubes_iterable
from punchbowl.exceptions import (
    CantInterpolateWarning,
    IncorrectPolarizationStateError,
    IncorrectTelescopeError,
    InvalidDataError,
)
from punchbowl.prefect import punch_flow, punch_task
from punchbowl.util import DataLoader, average_datetime, inpaint_nans, interpolate_data


class SkewFitResult:
    """Stores inputs and result of skewed Gaussian fitting."""

    def __init__(self, fit: MinimizerResult, bin_centers: np.ndarray, scaled_x_values: np.ndarray,
                 bin_values: np.ndarray, stack: np.ndarray, scale_factor: float, weights: np.ndarray) -> None:
        """Initialize class."""
        self.fit = fit
        self.bin_centers = bin_centers
        self.scaled_x_values = scaled_x_values
        self.bin_values = bin_values
        self.stack = stack
        self.scale_factor = scale_factor
        self.weights = weights
        self.x0 = self.fit.params["x0"].value
        self.A = self.fit.params["A"].value
        self.alpha = self.fit.params["alpha"].value
        self.sigma = self.fit.params["sigma"].value
        self.m = self.fit.params["m"].value
        self.b = self.fit.params["b"].value
        self.dx = np.min(np.diff(self.bin_centers))

    @cached_property
    def result(self) -> float:
        """Return the mode of the skewed Gaussian."""
        a = self.alpha

        # Find the mode of a skew Gaussian
        delta = a / np.sqrt(1 + a ** 2)
        mode = (
                np.sqrt(2 / np.pi) * delta
                - (1 - np.pi / 4) * (np.sqrt(2 / np.pi) * delta) ** 3 / (1 - 2 / np.pi * delta ** 2)
                - np.sign(a) / 2 * np.exp(-2 * np.pi / np.abs(a))
        )
        return (mode * self.sigma + self.x0) / self.scale_factor

    def fit_is_sus(self) -> bool:
        """Flag fits that look suspicious."""
        maxval = self.bin_values.max()
        if (3 * maxval < self.A
            or np.abs(self.alpha) > 10000
            or self.sigma < 0.5 * self.dx * self.scale_factor or self.sigma > 2 * (
            self.bin_centers[-1] - self.bin_centers[0]) * self.scale_factor
            or not (-4 * maxval < self.m * self.result * self.scale_factor + self.b < 2 * maxval)
        ):
            return True
        return any(param.value == param.min or param.value == param.max for param in self.fit.params.values())

    def plot(self, mark_result: bool = True) -> None:
        """Plot the fit."""
        import matplotlib.pyplot as plt  # noqa: PLC0415
        plt.step(self.bin_centers, self.bin_values, where="mid")
        plt.scatter(self.bin_centers, self.bin_values)

        fit_x = np.linspace(self.bin_centers[0], self.bin_centers[-1], 200)

        plt.plot(fit_x, skew_gaussian(fit_x * self.scale_factor, self.A, self.alpha, self.x0, self.sigma, 0, 0),
                 color="C1", label="Skew Gaussian")
        plt.plot(fit_x, skew_gaussian(fit_x * self.scale_factor, 0, 0, 0, 1, self.m, self.b), color="C2",
                 label="Linear")
        plt.plot(fit_x,
                 skew_gaussian(fit_x * self.scale_factor, self.A, self.alpha, self.x0, self.sigma, self.m, self.b),
                 color="C3", label="Model")
        if mark_result:
            plt.axvline(self.result, color="C4", label="Our result")
        plt.legend()


def skew_gaussian(x: np.ndarray, A: float, alpha: float, x0: float, sigma: float, m: float, b: float, # noqa: N803
                  ) -> np.ndarray:
    """Calculate a skewed Gaussian."""
    y = (x - x0) / sigma
    pdf = 1 / np.sqrt(2 * np.pi) * np.exp(-y ** 2 / 2)
    cdf = 1 / 2 * (1 + scipy.special.erf(alpha * y / np.sqrt(2)))
    return A * pdf * cdf + m * x + b


def _resid_skew(params: Parameters, scaled_x_values: np.ndarray, y_values: np.ndarray, bin_weights: np.ndarray,
                ) -> np.ndarray:
    """Evaluate function and return the residual."""
    params = params.valuesdict()
    resids = y_values - skew_gaussian(scaled_x_values, params["A"], params["alpha"], params["x0"], params["sigma"],
                                      params["m"], params["b"])
    resids *= bin_weights
    return resids


def pick_peak(bin_values: np.ndarray) -> int:
    """Pick the first (left-most) peak, but isn't fooled if the bins dip by <20% and then keep going up."""
    largest_seen = -1
    largest_idx = -1
    i = 0
    while True:
        if bin_values[i] > largest_seen:
            largest_seen = bin_values[i]
            largest_idx = i
        if bin_values[i] < 0.8 * largest_seen:
            return largest_idx
        i += 1
        if i >= len(bin_values):
            return largest_idx


def fit_skew(stack: np.ndarray, ret_all: bool = False, x_scale_factor: float = 1e13, weight: bool = True, # noqa: C901
             plot_histogram_steps: bool = False) -> float | SkewFitResult:
    """Fit a skewed Gaussiain to a histrogram of data values to estimate the stray light value."""
    # Start by trimming outliers
    low, high = np.nanpercentile(stack, (0.04, 99))
    if plot_histogram_steps:
        import matplotlib.pyplot as plt  # noqa: PLC0415
        plt.hist(stack, bins=50, range=(low, high))

    # We make a relatively fine histogram so we can pick out the peak region and zoom in there
    bin_values, bin_edges, *_ = np.histogram(stack, bins=50, range=(low, high))
    dx = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1] + dx / 2

    # From the highest bin, work out until we find a bin that has dropped by more than a certain amount. That will set
    # our zoom-in range
    imax = pick_peak(bin_values)
    peak_val = bin_values[imax]
    bmax = bin_values[imax]
    for istart in range(imax - 1, -1, -1):
        if bin_values[istart] < .4 * bmax:
            break
    else:
        istart = 0
    low = bin_edges[istart]

    for istop in range(imax, len(bin_values)):
        if bin_values[istop] < .5 * bmax:
            break
    if istop > len(bin_values) - 1:
        istop = len(bin_values) - 1
    high = bin_edges[1 + istop]

    # Now we'll zoom in to that region, make a histogram, and then if the peak doesn't seem wide enough (in terms of
    # number of bins), we'll zoom in further
    while True:
        if plot_histogram_steps:
            plt.axvline(low)
            plt.axvline(high)
            plt.show()
        if np.sum((stack > low) * (stack < high)) < 30:
            break
        bin_values, bin_edges, *_ = np.histogram(stack, bins=20, range=(low, high))
        dx = bin_edges[1] - bin_edges[0]
        bin_centers = bin_edges[:-1] + dx / 2

        # Sometimes there are empty bins that just seem to make the fit worse, so exclude them
        full_bins = bin_values > .05 * bin_values.max()
        bin_values = bin_values[full_bins]
        bin_centers = bin_centers[full_bins]
        # No longer valid
        del bin_edges

        # Don't just take the largest bin as the peak---occasionally there are two peaks in the range, and we want the
        # one that's dimmer (i.e. pixel values that are lower), not the peak that's higher on our histogram
        imax = pick_peak(bin_values)
        peak_val = bin_values[imax]

        # See how many bins wide our peak is (roughly)
        p2p = peak_val - np.min(bin_values)
        # We're comparing bins' height above the minimum, not the height above 0!
        n_in_peak = np.sum(bin_values > peak_val - 0.4 * p2p)
        if plot_histogram_steps:
            plt.hist(stack, range=(low, high), bins=20)
            plt.title(f"p2p {p2p}, thresh {peak_val - 0.4 * p2p}, n {n_in_peak}")
        if n_in_peak > 0.2 * len(bin_values):
            break

        center = bin_centers[imax]
        dlow = center - low
        low = center - 0.75 * dlow
        dhigh = high - center
        high = center + 0.75 * dhigh
    if plot_histogram_steps:
        plt.show()

    bin_weights = 1 / (40 + np.abs(np.arange(0, len(bin_values)) - imax))
    bin_weights /= bin_weights.max()

    if not weight:
        bin_weights = np.ones_like(bin_weights)

    params = Parameters()
    params.add("A", value=np.max(bin_values), min=0, max=2 * peak_val)
    params.add("alpha", value=0, min=0)
    params.add("x0",
               value=x_scale_factor * bin_centers[np.argmax(bin_values)],
               min=(bin_centers[0] - dx) * x_scale_factor,
               max=(bin_centers[-1] + dx) * x_scale_factor)
    params.add("sigma", value=6 * dx * x_scale_factor, min=1e-20, max=10)
    params.add("m", value=0, vary=True)
    params.add("b", value=0, vary=True)

    scaled_x_values = bin_centers * x_scale_factor

    with np.errstate(all="ignore"):
        out = minimize(_resid_skew, params, args=(scaled_x_values, bin_values, bin_weights), method="least_squares",
                       calc_covar=False)

    r = SkewFitResult(out, bin_centers=bin_centers, scaled_x_values=scaled_x_values, bin_values=bin_values,
                      stack=stack, scale_factor=x_scale_factor, weights=bin_weights)

    if ret_all:
        return r
    if r.fit_is_sus():
        return np.nan
    return r.result


def _estimate_stray_light_one_slice(data_slice: np.ndarray, x_grid: np.ndarray, half_width: int) -> np.ndarray:
    result = np.empty(x_grid.shape)
    for j in range(len(x_grid)):
        x = x_grid[j]
        stack = data_slice[:, :, x - half_width:x + half_width + 1]
        n_pts = stack.size
        stack = stack[stack > 1e-15]
        if stack.size < n_pts / 2: # noqa: SIM108
            r = 0
        else:
            r = fit_skew(stack, False)
        result[j] = r
    return result


@punch_flow
def estimate_stray_light(filepaths: list[str], # noqa: C901
                         do_uncertainty: bool = True,
                         reference_time: datetime | str | None = None,
                         stride: int = 1,
                         window_size: int = 3,
                         num_workers: int | None = None,
                         num_loaders: int | None = None) -> list[NDCube]:
    """Estimate the fixed stray light pattern using a percentile."""
    logger = get_run_logger()
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd")
    logger.info(f"Running with {len(filepaths)} input files")
    if isinstance(reference_time, str):
        reference_time = datetime.strptime(reference_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    data = None
    uncertainty = None
    date_obses = []
    n_failed = 0
    j = 0
    if isinstance(filepaths[0], NDCube):
        iterator = filepaths
    else:
        iterator = load_many_cubes_iterable(filepaths, n_workers=num_loaders, allow_errors=True)
    for i, result in enumerate(iterator):
        if isinstance(result, str):
            logger.warning(f"Loading {filepaths[i]} failed")
            logger.warning(result)
            n_failed += 1
            if n_failed > 10:
                raise RuntimeError(f"{n_failed} files failed to load, stopping")
            continue
        cube = result
        date_obses.append(cube.meta.datetime)
        if data is None:
            data = np.empty((len(filepaths), *cube.data.shape))
        data[j] = cube.data
        j += 1
        if do_uncertainty:
            if uncertainty is None:
                uncertainty = np.zeros_like(cube.data)
            if cube.uncertainty is not None:
                # The final uncertainty is sqrt(sum(square(input uncertainties))), so we accumulate the squares here
                uncertainty += cube.uncertainty.array ** 2
        if (i+1) % 50 == 0:
            logger.info(f"Loaded {i+1}/{len(filepaths)} files")
    # Crop the unused end of the array if we had a few files that errored out
    data = data[:j+1]

    logger.info(f"Images loaded; they span {min(date_obses).strftime('%Y-%m-%dT%H:%M:%S')} to "
                f"{max(date_obses).strftime('%Y-%m-%dT%H:%M:%S')}")

    window_half_width = window_size // 2
    x_grid = np.arange(window_half_width, data.shape[2] - window_half_width, stride)
    y_grid = np.arange(window_half_width, data.shape[1] - window_half_width, stride)
    def args() -> Generator[tuple]:
        # Build a grid with `stride` as the spacing, but exclude from the edges so that the window we use at each
        # stride position fits
        for i in range(window_half_width, data.shape[1] - window_half_width, stride):
            data_slice = data[:, i-window_half_width:i+window_half_width+1, :]
            yield data_slice, x_grid, window_half_width

    ctx = multiprocessing.get_context("forkserver")
    with ctx.Pool(num_workers) as p:
        stray_light_estimate = np.stack(p.starmap(_estimate_stray_light_one_slice, args()), axis=0)

    stray_light_estimate = inpaint_nans(stray_light_estimate, kernel_size=5)
    if stride > 1 or window_size > 1:
        interper = scipy.interpolate.RegularGridInterpolator(
                (y_grid, x_grid), stray_light_estimate, method="linear", bounds_error=False, fill_value=None)
        out_y, out_x = np.mgrid[:data.shape[1], :data.shape[2]]
        stray_light_estimate = interper(np.stack((out_y, out_x), axis=-1))

    uncertainty = np.sqrt(uncertainty) / len(filepaths) if do_uncertainty else None

    out_type = "S" + cube.meta.product_code[1:]
    meta = NormalizedMetadata.load_template(out_type, "1")
    meta["DATE-AVG"] = average_datetime(date_obses).strftime("%Y-%m-%dT%H:%M:%S")
    meta["DATE-OBS"] = reference_time.strftime("%Y-%m-%dT%H:%M:%S") if reference_time else meta["DATE-AVG"].value
    meta["DATE-BEG"] = min(date_obses).strftime("%Y-%m-%dT%H:%M:%S")
    meta["DATE-END"] = max(date_obses).strftime("%Y-%m-%dT%H:%M:%S")
    meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    meta.history.add_now("stray light",
                         f"Generated with {len(filepaths)} files running from "
                         f"{min(date_obses).strftime('%Y-%m-%dT%H:%M:%S')} to "
                         f"{max(date_obses).strftime('%Y-%m-%dT%H:%M:%S')}")
    meta["FILEVRSN"] = cube.meta["FILEVRSN"].value

    # Let's put in a valid, representative WCS, with the right scale and pointing, etc.
    wcs = cube.wcs
    out_cube = NDCube(data=stray_light_estimate, meta=meta, wcs=wcs, uncertainty=StdDevUncertainty(uncertainty))

    return [out_cube]

@punch_flow
def estimate_polarized_stray_light(
                mfilepaths: list[str],
                zfilepaths: list[str],
                pfilepaths: list[str],
                do_uncertainty: bool = True,
                reference_time: datetime | str | None = None,
                stride: int = 1,
                window_size: int = 3,
                num_workers: int | None = None,
                num_loaders: int | None = None,
                ) -> list[NDCube]:
    """Estimate the polarized stray light pattern using minimum indexing method."""
    logger = get_run_logger()

    if isinstance(reference_time, str):
        reference_time = datetime.strptime(reference_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)

    output_cubes = []
    logger.info("Running for M files")
    output_cubes.extend(estimate_stray_light(mfilepaths,
                                             do_uncertainty=do_uncertainty,
                                             reference_time=reference_time,
                                             stride=stride,
                                             window_size=window_size,
                                             num_workers=num_workers,
                                             num_loaders=num_loaders))
    logger.info("Running for Z files")
    output_cubes.extend(estimate_stray_light(zfilepaths,
                                             do_uncertainty=do_uncertainty,
                                             reference_time=reference_time,
                                             stride=stride,
                                             window_size=window_size,
                                             num_workers=num_workers,
                                             num_loaders=num_loaders))
    logger.info("Running for P files")
    output_cubes.extend(estimate_stray_light(pfilepaths,
                                             do_uncertainty=do_uncertainty,
                                             reference_time=reference_time,
                                             stride=stride,
                                             window_size=window_size,
                                             num_workers=num_workers,
                                             num_loaders=num_loaders))
    return output_cubes

@punch_task
def remove_stray_light_task(data_object: NDCube, #noqa: C901
                            stray_light_before_path: pathlib.Path | str | NDCube | DataLoader,
                            stray_light_after_path: pathlib.Path | str | NDCube | DataLoader) -> NDCube:
    """
    Prefect task to remove stray light from an image.

    Stray light is light in an optical system which was not intended in the
    design.

    The PUNCH instrument stray light will be mapped periodically as part of the
    ongoing in-flight calibration effort. The stray light maps will be
    generated directly from the L0 and L1 science data. Separating instrumental
    stray light from the F-corona. This has been demonstrated with SOHO/LASCO
    and with STEREO/COR2 observations. It requires an instrumental roll to hold
    the stray light pattern fixed while the F-corona rotates in the field of
    view. PUNCH orbital rolls will be used to create similar effects.

    Uncertainty across the image plane is calculated using a known stray light
    model and the difference between the calculated stray light and the ground
    truth. The uncertainty is convolved with the input uncertainty layer to
    produce the output uncertainty layer.


    Parameters
    ----------
    data_object : NDCube
        data to operate on

    stray_light_before_path: pathlib
        path to stray light model before observation to apply to data

    stray_light_after_path: pathlib
        path to stray light model after observation to apply to data

    Returns
    -------
    NDCube
        modified version of the input with the stray light removed

    """
    if stray_light_before_path is None or stray_light_after_path is None:
        data_object.meta.history.add_now("LEVEL1-remove_stray_light", "Stray light correction skipped")
        return data_object

    if isinstance(stray_light_before_path, NDCube):
        stray_light_before_model = stray_light_before_path
    elif isinstance(stray_light_before_path, DataLoader):
        stray_light_before_model = stray_light_before_path.load()
    else:
        stray_light_before_path = pathlib.Path(stray_light_before_path)
        if not stray_light_before_path.exists():
            msg = f"File {stray_light_before_path} does not exist."
            raise InvalidDataError(msg)
        stray_light_before_model = load_ndcube_from_fits(stray_light_before_path)

    if isinstance(stray_light_after_path, NDCube):
        stray_light_after_model = stray_light_after_path
    elif isinstance(stray_light_after_path, DataLoader):
        stray_light_after_model = stray_light_after_path.load()
    else:
        stray_light_after_path = pathlib.Path(stray_light_after_path)
        if not stray_light_after_path.exists():
            msg = f"File {stray_light_after_path} does not exist."
            raise InvalidDataError(msg)
        stray_light_after_model = load_ndcube_from_fits(stray_light_after_path)

    for model in stray_light_before_model, stray_light_after_model:
        if model.meta["TELESCOP"].value != data_object.meta["TELESCOP"].value:
            msg=f"Incorrect TELESCOP value within {model['FILENAME'].value}"
            raise IncorrectTelescopeError(msg)
        if model.meta["OBSLAYR1"].value != data_object.meta["OBSLAYR1"].value:
            msg=f"Incorrect polarization state within {model['FILENAME'].value}"
            raise IncorrectPolarizationStateError(msg)
        if model.data.shape != data_object.data.shape:
            msg = f"Incorrect stray light function shape within {model['FILENAME'].value}"
            raise InvalidDataError(msg)

    # For the quickpunch case, our stray light models run right up to the current time, with their DATE-OBS likely days
    # in the past. It feels reckless to interpolate the six-hour variation in the model over several days, so let's
    # instead interpolate using the nearst of DATE-BEG, DATE-AVG, or DATE-END. (DATE-BEG will be the best choice when
    # reprocessing.)
    delta_dateavg = abs(parse_datetime(stray_light_before_model.meta["DATE-AVG"].value + " UTC")
                        - data_object.meta.datetime)
    delta_datebeg = abs(parse_datetime(stray_light_before_model.meta["DATE-BEG"].value + " UTC")
                        - data_object.meta.datetime)
    delta_dateend = abs(parse_datetime(stray_light_before_model.meta["DATE-END"].value + " UTC")
                        - data_object.meta.datetime)

    closest = min(delta_datebeg, delta_dateavg, delta_dateend)
    if closest is delta_datebeg:
        time_key = "DATE-BEG"
    elif closest is delta_dateavg:
        time_key = "DATE-AVG"
    else:
        time_key = "DATE-END"

    if stray_light_before_model.meta[time_key].value == stray_light_after_model.meta[time_key].value:
        warnings.warn(
            "Timestamps are identical for the stray light models; can't inter/extrapolate", CantInterpolateWarning)
        stray_light_model = stray_light_before_model.data
    else:
        stray_light_model = interpolate_data(stray_light_before_model,
                                             stray_light_after_model,
                                             data_object.meta.datetime,
                                             time_key=time_key,
                                             allow_extrapolation=True)
    data_object.data[:, :] -= stray_light_model
    uncertainty = 0
    # TODO: when we have real uncertainties, use them
    # uncertainty = stray_light_model.uncertainty.array # noqa: ERA001
    data_object.uncertainty.array[...] = np.sqrt(data_object.uncertainty.array**2 + uncertainty**2)
    data_object.meta.history.add_now("LEVEL1-remove_stray_light",
                                     f"stray light removed with {stray_light_before_model.meta['FILENAME'].value} "
                                     f"and {stray_light_after_model.meta['FILENAME'].value}")
    return data_object
