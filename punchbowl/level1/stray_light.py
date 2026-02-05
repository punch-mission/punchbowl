import pathlib
import warnings
import multiprocessing
from datetime import UTC, datetime
from functools import cached_property
from itertools import pairwise
from collections.abc import Generator

import numpy as np
import scipy
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from dateutil.parser import parse as parse_datetime
from lmfit import Parameters, minimize
from lmfit.minimizer import MinimizerResult
from ndcube import NDCube
from prefect import get_run_logger
from skimage.restoration import inpaint

from punchbowl.data import NormalizedMetadata, load_ndcube_from_fits
from punchbowl.data.punch_io import load_many_cubes_iterable
from punchbowl.exceptions import (
    CantInterpolateWarning,
    IncorrectPolarizationStateError,
    IncorrectTelescopeError,
    InvalidDataError,
)
from punchbowl.prefect import punch_flow, punch_task
from punchbowl.util import (
    DataLoader,
    average_datetime,
    inpaint_nans,
    interpolate_data,
    load_mask_file,
    nan_gaussian,
    parallel_sort_first_axis,
)


class SkewFitResult:
    """Stores inputs and result of skewed Gaussian fitting."""

    def __init__(self, fit: MinimizerResult, bin_centers: np.ndarray, scaled_x_values: np.ndarray,
                 bin_values: np.ndarray, stack: np.ndarray, scale_factor: float, weights: np.ndarray,
                 target_center: float) -> None:
        """Initialize class."""
        self.fit = fit
        self.bin_centers = bin_centers
        self.scaled_x_values = scaled_x_values
        self.bin_values = bin_values
        self.stack = stack
        self.scale_factor = scale_factor
        self.weights = weights
        self.target_center = target_center
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
        # Uses the approximation from https://en.wikipedia.org/wiki/Skew_normal_distribution. I took the factor of
        # 1/np.sqrt(2) out of the actual skew-gaussian function, so what we have as the fitted alpha is *actually*
        # alpha / sqrt(2) # noqa: ERA001
        a = self.alpha * np.sqrt(2)

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

    def plot(self, mark_result: bool = True, mark_tcenter: bool = True) -> None:
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
        if mark_tcenter:
            plt.axvline(self.target_center, color='C5', label="Targeted peak", ls=':')
        plt.legend()


# To compute skew-gaussians a bit faster, we pre-generate a lookup table. This table's resolution is good to within
# 0.0002 (where the absolute values are within [0, 1]), and where the function values aren't minute, the table is
# good to within 0.02%. Using the table saves about 30% of the computation time---and we compute a *lot* of skew
# Gaussians!
pdf_table_vals = np.arange(-4.2, 4.2, 0.02)
cdf_table_vals = np.arange(-3, 3, 0.02)
pdf_vals = np.exp(-0.5 * pdf_table_vals**2)
cdf_vals = 1 + scipy.special.erf(cdf_table_vals)
def skew_gaussian(x: np.ndarray, A: float, alpha: float, x0: float, sigma: float, m: float, b: float, # noqa: N803
                  ) -> np.ndarray:
    """Calculate a skewed Gaussian."""
    y = (x - x0) / sigma
    pdf = np.interp(y, pdf_table_vals, pdf_vals, left=0, right=0)
    cdf = np.interp(alpha * y, cdf_table_vals, cdf_vals, left=0, right=2)
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
    peak_min_height = 0.15 * bin_values.max()
    largest_seen = -1
    largest_idx = -1
    n_consecutive_downhill = 0
    i = 0
    while True:
        if i > 0:
            if bin_values[i] >= bin_values[i-1]:
                n_consecutive_downhill = 0
            else:
                n_consecutive_downhill += 1
        if bin_values[i] > largest_seen:
            largest_seen = bin_values[i]
            largest_idx = i
        else:
            peak_seems_peakish = bin_values[i] < 0.85 * largest_seen or n_consecutive_downhill >= 3
            peak_is_valid = largest_seen >= peak_min_height
            this_bin_is_ok = bin_values[i] > 0
            if peak_seems_peakish and peak_is_valid and this_bin_is_ok:
                return largest_idx
        i += 1
        if i >= len(bin_values):
            return largest_idx


def find_peak_end(bin_values: np.ndarray, peak_location, direction) -> int:
    lowest_seen = np.inf
    lowest_idx = -1
    i = peak_location
    while True:
        if bin_values[i] < lowest_seen:
            lowest_seen = bin_values[i]
            lowest_idx = i
        if bin_values[i] > 1.25 * lowest_seen and bin_values[i] > 0:
            return lowest_idx
        i += direction
        if i >= len(bin_values) or i < 0:
            return lowest_idx


class OutOfPointsError(RuntimeError):
    pass


def fit_skew(stack: np.ndarray, ret_all: bool = False, x_scale_factor: float = 1e13, weight: bool = True, # noqa: C901
             plot_histogram_steps: bool = False, exclude_above_percentile: float = 0) -> float | SkewFitResult:
    """Fit a skewed Gaussian to a histogram of data values to estimate the stray light value."""
    # Start by trimming outliers. We do that by making a histogram, finding the tallest bin, and then working out
    # from there until we hit bins with little to no counts. We care about the main part of the distribution,
    # so anything beyond those (nearly-) empty bins is an outlier. So we exclude those points and "zoom in" by
    # re-making the histogram using only points between those two identified bins. This process repeats until there
    # aren't any (nearly-) empty bins.
    if exclude_above_percentile:
        percentile_value = np.percentile(stack, exclude_above_percentile)
        stack = stack[stack < percentile_value]

    bin_values, bin_edges, *_ = np.histogram(stack, bins=50)
    dx = bin_edges[1] - bin_edges[0]
    if plot_histogram_steps:
        import matplotlib.pyplot as plt  # noqa: PLC0415
        dx = bin_edges[1] - bin_edges[0]
        plt.bar(bin_edges[:-1] + dx/2, bin_values, width=dx)

    min_count = 0.01 * bin_values.max()
    # Safety valve to avoid infinite loops
    max_loops_remaining = 10
    while np.any(bin_values <= min_count) and max_loops_remaining:
        max_loops_remaining -= 1
        peak = np.argmax(bin_values)

        # Go to the left, looking for nearly-empty bins
        istart = peak
        while bin_values[istart] > min_count and istart > 0:
            istart -= 1
        # Set our new low bound to be the high edge of the bin if it's empty, or the low end if it's full but it's
        # the last bin.
        stopped_on_small_bin = istart > 0 or bin_values[istart] <= min_count
        low = bin_edges[istart + 1] if stopped_on_small_bin else bin_edges[istart]

        # Go to the right, looking for nearly-empty bins
        istop = peak
        while bin_values[istop] > min_count and istop < len(bin_values) - 1:
            istop += 1
        stopped_on_small_bin = istop < len(bin_values) - 1 or bin_values[istop] <= min_count
        high = bin_edges[istop] if stopped_on_small_bin else bin_edges[istop + 1]

        if plot_histogram_steps:
            plt.axvline(low)
            plt.axvline(high)
            plt.title("Zooming to cut outlier bins")
            plt.show()

        # Re-make the histogram within these bounds
        bin_values, bin_edges, *_ = np.histogram(stack, bins=50, range=(low, high))
        dx = bin_edges[1] - bin_edges[0]
        if plot_histogram_steps:
            plt.bar(bin_edges[:-1] + dx/2, bin_values, width=dx)

        if np.sum(bin_values) < 100:
            raise OutOfPointsError()
        min_count = 0.01 * bin_values.max()


    # Now the outliers should be gone. When present, they were dragging the range of our histogram way out,
    # so the core distribution had very poor resolution. Now we should have good resolution on the core area,
    # and we can refine our zoom range better. Next we identify the target peak, walk downhill from it to find its
    # edges, and we zoom there to isolate our targeted peak and avoid fitting a different peak
    imax = pick_peak(bin_values)
    peak_location = bin_edges[imax] + dx / 2
    ilow = find_peak_end(bin_values, imax, -1)
    ihigh = find_peak_end(bin_values, imax, 1)
    low = bin_edges[ilow]
    high = bin_edges[ihigh + 1]
    if plot_histogram_steps:
        plt.axvline(peak_location, ls='--')
        plt.axvline(low)
        plt.axvline(high)
        plt.title("Zooming in to isolate peak")
        plt.show()

    bin_values, bin_edges, *_ = np.histogram(stack, bins=20, range=(low, high))
    dx = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1] + dx / 2

    if plot_histogram_steps:
        plt.bar(bin_edges[:-1] + dx/2, bin_values, width=dx)

    if np.sum(bin_values) < 100:
        raise OutOfPointsError()

    # Next we walk out from the target peak until we find binds that are low relative to the peak, to chop off the
    # tails of the distribution.
    imax = pick_peak(bin_values)
    peak_val = bin_values[imax]
    for istart in range(imax - 1, -1, -1):
        if bin_values[istart] < .4 * peak_val:
            break
    else:
        istart = 0
    low = bin_edges[istart]

    for istop in range(imax, len(bin_values)):
        if bin_values[istop] < .5 * peak_val:
            break
    if istop > len(bin_values) - 1:
        istop = len(bin_values) - 1
    high = bin_edges[1 + istop]

    if plot_histogram_steps:
        plt.title("Zooming in to exclude tail")
        plt.axvline(bin_edges[imax] + dx/2, ls='--')
        plt.axvline(low)
        plt.axvline(high)
        plt.show()

    bin_values, bin_edges, *_ = np.histogram(stack, bins=20, range=(low, high))
    if np.sum(bin_values) < 100:
        raise OutOfPointsError()
    dx = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1] + dx / 2
    imax = pick_peak(bin_values)
    peak_val = bin_values[imax]
    peak_location = bin_centers[imax]
    if plot_histogram_steps:
        plt.bar(bin_centers, bin_values, width=dx)

    # Now we'll zoom in to that region, make a histogram, and then if the peak doesn't seem wide enough (in terms of
    # number of bins), we'll zoom in further
    while True:
        # We need to compute how many bins wide our peak is (roughly)
        p2p = peak_val - np.min(bin_values)
        # We're comparing bins' height above the minimum bin value, not the height above 0!
        n_in_peak = np.sum(bin_values > peak_val - 0.4 * p2p)
        if plot_histogram_steps:
            plt.suptitle(f"p2p {p2p}, thresh {peak_val - 0.4 * p2p}, {n_in_peak} bins above thresh")
        if n_in_peak > 0.2 * len(bin_values):
            break

        center = bin_centers[imax]
        dlow = center - low
        low = center - 0.8 * dlow
        dhigh = high - center
        high = center + 0.8 * dhigh

        if plot_histogram_steps:
            plt.title("Zooming in to widen peak")
            plt.axvline(bin_edges[imax] + dx/2, ls='--')
            plt.axvline(low)
            plt.axvline(high)
            plt.show()

        bin_values, bin_edges, *_ = np.histogram(stack, bins=20, range=(low, high))
        if np.sum(bin_values) < 100:
            raise OutOfPointsError()
        dx = bin_edges[1] - bin_edges[0]
        bin_centers = bin_edges[:-1] + dx / 2
        if plot_histogram_steps:
            plt.bar(bin_centers, bin_values, width=dx)
        imax = pick_peak(bin_values)
        peak_val = bin_values[imax]
        peak_location = bin_centers[imax]

    if plot_histogram_steps:
        plt.axvline(peak_location, ls='--')
        plt.title("Final distribution")
        plt.show()

    # Sometimes there are empty bins that just seem to make the fit worse, so exclude them
    full_bins = bin_values > .05 * peak_val
    bin_values = bin_values[full_bins]
    if np.sum(bin_values) < 100 or len(bin_values) < 5:
        raise OutOfPointsError()
    bin_centers = bin_centers[full_bins]
    imax = np.where(bin_values == peak_val)[0][0]
    # No longer valid
    del bin_edges

    if weight:
        bin_weights = 1 / (40 + np.abs(np.arange(0, len(bin_values)) - imax))
        bin_weights /= bin_weights.max()
    else:
        bin_weights = np.ones_like(bin_values)

    params = Parameters()
    params.add("A", value=0.5/np.sqrt(2*np.pi) * np.max(bin_values), min=0, max=2 * peak_val)
    params.add("alpha", value=0, min=0)
    params.add("x0",
               value=x_scale_factor * peak_location,
               min=(bin_centers[0] - dx) * x_scale_factor,
               max=(bin_centers[-1] + dx) * x_scale_factor)
    params.add("sigma", value=6 * dx * x_scale_factor, min=1e-20, max=10)
    params.add("m", value=0, vary=True)
    params.add("b", value=0, vary=True)

    scaled_x_values = bin_centers * x_scale_factor

    with np.errstate(all="ignore"):
        out = minimize(_resid_skew, params, args=(scaled_x_values, bin_values, bin_weights), method="least_squares",
                       calc_covar=False, ftol=2e-4, gtol=2e-4)

    r = SkewFitResult(out, bin_centers=bin_centers, scaled_x_values=scaled_x_values, bin_values=bin_values,
                      stack=stack, scale_factor=x_scale_factor, weights=bin_weights, target_center=peak_location)

    if ret_all:
        return r
    if r.fit_is_sus():
        return np.nan
    return r.result


REQUIRED_FRACTION_OF_NEIGHBORHOOD_PIXELS = 0.5


def _estimate_stray_light_one_slice(y: int, data_slice: np.ndarray, x_grid: np.ndarray, half_width: int) -> np.ndarray:
    noise_mode = 2e-13
    noise_hwhm = 2.25e-13 - 1.8e-13
    noise_amp = 0.2

    result = np.empty(x_grid.shape)
    for j, x in enumerate(x_grid):
        stack = data_slice[:, :, x - half_width:x + half_width + 1]
        n_pts = stack.size
        stack = stack[stack > 1e-15]
        if stack.size < n_pts * REQUIRED_FRACTION_OF_NEIGHBORHOOD_PIXELS:
            r = np.nan
        else:
            rng = np.random.default_rng(y * x + y + x)
            noise = noise_amp * rng.normal(scale=np.sqrt(stack / noise_mode) * noise_hwhm, size=stack.size)
            stack += noise
            try:
                r = fit_skew(stack, False)
            except OutOfPointsError:
                r = np.nan
        result[j] = r
    return result


@punch_flow
def estimate_stray_light(filepaths: list[str], # noqa: C901
                         do_uncertainty: bool = True,
                         reference_time: datetime | str | None = None,
                         stride: int = 10,
                         window_size: int = 5,
                         blur_sigma: float = 1.5,
                         n_crota_bins: int = 30,
                         crota_bin_width: float = 45,
                         image_mask_path: str | None = None,
                         num_workers: int | None = None,
                         num_loaders: int | None = None) -> list[NDCube]:
    """Estimate the fixed stray light pattern using a percentile."""
    logger = get_run_logger()
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd")
    logger.info(f"Running with {len(filepaths)} input files")
    if isinstance(reference_time, str):
        reference_time = datetime.strptime(reference_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)

    image_mask = load_mask_file(image_mask_path) if image_mask_path is not None else None
    strided_image_mask = None

    data_array = None
    uncertainty = None
    metas = []
    j = 0
    n_failed = 0
    logger.info(f"Will read {len(filepaths)} images")
    if isinstance(filepaths[0], NDCube):
        iterable = filepaths
    else:
        iterable = load_many_cubes_iterable(filepaths, n_workers=num_loaders, allow_errors=True,
                                            include_uncertainty=do_uncertainty,
                                            include_provenance=False, dtype=np.float32)
    for i, result in enumerate(iterable):
        if isinstance(result, str):
            logger.warning(f"Loading {filepaths[i]} failed")
            logger.warning(result)
            n_failed += 1
            if n_failed > 10:
                raise RuntimeError(f"{n_failed} files failed to load, stopping")
            continue
        # We need to save a sample cube (not a string/error message) for the end of this flow
        cube = result
        if data_array is None:
            data_array = np.empty((len(filepaths), *cube.data.shape), dtype=cube.data.dtype)
        data_array[j] = cube.data
        j += 1
        metas.append(cube.meta)

        if do_uncertainty:
            if uncertainty is None:
                uncertainty = np.zeros_like(cube.data)
            if cube.uncertainty is not None:
                # The final uncertainty is sqrt(sum(square(input uncertainties))), so we accumulate the squares here
                uncertainty += np.nan_to_num(cube.uncertainty.array, posinf=0, neginf=0) ** 2
        if (i + 1) % 100 == 0:
            logger.info(f"Loaded {i + 1}/{len(filepaths)} files")
    logger.info("Finished loaded files")
    data_array = data_array[:j]

    if image_mask is None:
        image_mask = np.all(data_array == 0, axis=0)

    bin_centers = np.linspace(-180, 180, n_crota_bins, endpoint=False)
    bin_starts = bin_centers - crota_bin_width / 2
    bin_stops = bin_centers + crota_bin_width / 2

    crota_is_in_bin = (lambda crota, binn: ((bin_starts[binn] < crota <= bin_stops[binn])
                                         or (bin_starts[binn] < crota - 360 <= bin_stops[binn])
                                         or (bin_starts[binn] < crota + 360 <= bin_stops[binn])))

    bin_masks = []
    for binn in range(n_crota_bins):
        mask = np.array([crota_is_in_bin(m['CROTA'].value, binn) for m in metas])
        bin_masks.append(mask)

    models = []
    for bin_n, bin_mask in enumerate(bin_masks):
        logger.info(f"Starting bin {bin_n + 1}")

        window_half_width = window_size // 2
        # Build a grid with `stride` as the spacing, but exclude from the edges so that the window we use at each
        # stride position fits
        x_grid = np.arange(window_half_width, data_array.shape[2] - window_half_width, stride)
        y_grid = np.arange(window_half_width, data_array.shape[1] - window_half_width, stride)

        if strided_image_mask is None:
            strided_image_mask = np.empty((y_grid.size, x_grid.size), dtype=bool)
            for i, y in enumerate(y_grid):
                for j, x in enumerate(x_grid):
                    sample = image_mask[y - window_half_width:y + window_half_width + 1,
                                        x - window_half_width:x + window_half_width + 1]
                    strided_image_mask[i, j] = sample.sum() > sample.size * REQUIRED_FRACTION_OF_NEIGHBORHOOD_PIXELS

        def args() -> Generator[tuple]:
            for y in y_grid:
                data_slice = data_array[bin_mask, y - window_half_width:y + window_half_width + 1, :]
                yield y, data_slice, x_grid, window_half_width

        logger.info("Beginning model fitting")
        ctx = multiprocessing.get_context("forkserver")
        with ctx.Pool(num_workers) as p:
            stray_light_estimate = np.stack(p.starmap(_estimate_stray_light_one_slice, args()), axis=0)
        stray_light_estimate[~strided_image_mask] = 0
        logger.info("Finished model fitting")

        # import matplotlib.pyplot as plt
        # plt.imshow(stray_light_estimate, vmin=0, vmax=.5e-12, origin='lower')
        # plt.title("Raw")
        # plt.show()

        stray_light_estimate = inpaint_nans(stray_light_estimate, kernel_size=5)
        # plt.imshow(stray_light_estimate, vmin=0, vmax=.5e-12, origin='lower')
        # plt.title("post inpaint")
        # plt.show()

        d = data_array[:, y_grid][:, :, x_grid]
        d = parallel_sort_first_axis(d, inplace=True)
        percentiles = np.argmin(np.abs(d - stray_light_estimate), axis=0) / d.shape[0] * 100
        # plt.imshow(percentiles, vmin=0, vmax=80, origin='lower')
        # plt.title("Percentiles")
        # plt.show()
        del d
        bad_region = percentiles >= 70
        bad_region = scipy.ndimage.binary_fill_holes(bad_region)
        bad_region = scipy.ndimage.binary_opening(bad_region, iterations=int(np.ceil(2*stride/10)))
        bad_region = scipy.ndimage.binary_dilation(bad_region, iterations=int(np.ceil(8*stride/10)))
        bad_region *= strided_image_mask

        # plt.imshow(bad_region, vmin=0, vmax=1, origin='lower')
        # plt.title("bad region")
        # plt.show()

        inpaint_mask = bad_region + ~strided_image_mask
        inpainted = inpaint.inpaint_biharmonic(stray_light_estimate, inpaint_mask)
        # plt.imshow(inpainted, vmin=0, vmax=.5e-12, origin='lower')
        # plt.title("Inpainted")
        # plt.show()

        stray_light_estimate[bad_region] = inpainted[bad_region]

        stray_light_estimate[~strided_image_mask] = np.nan

        # plt.imshow(stray_light_estimate, vmin=0, vmax=.5e-12, origin='lower')
        # plt.title("Filled")
        # plt.show()

        if blur_sigma:
            stray_light_estimate = nan_gaussian(stray_light_estimate, blur_sigma)
        # plt.imshow(stray_light_estimate, vmin=0, vmax=.5e-12, origin='lower')
        # plt.title("Blurred")
        # plt.show()

        stray_light_estimate[~strided_image_mask] = 0

        # plt.imshow(stray_light_estimate, vmin=0, vmax=.5e-12, origin='lower')
        # plt.title("Masked")
        # plt.show()
        if stride > 1 or window_size > 1:
            interper = scipy.interpolate.RegularGridInterpolator(
                    (y_grid, x_grid), stray_light_estimate, method="linear", bounds_error=False, fill_value=None)
            out_y, out_x = np.mgrid[:data_array.shape[1], :data_array.shape[2]]
            stray_light_estimate = interper(np.stack((out_y, out_x), axis=-1))
            stray_light_estimate *= image_mask
        # plt.imshow(stray_light_estimate, vmin=0, vmax=.5e-12, origin='lower')
        # plt.title("Interped, final")
        # plt.show()

        models.append(stray_light_estimate)
        logger.info(f"Finished with bin {bin_n + 1}")

    del data_array

    if do_uncertainty:
        uncertainty = np.sqrt(uncertainty) / len(filepaths)

    out_type = "S" + metas[0].product_code[1:]
    meta = NormalizedMetadata.load_template(out_type, "1")
    meta.provenance = [m['FILENAME'] for m in metas]
    all_date_obses = [m.datetime for m in metas]
    meta["DATE-AVG"] = average_datetime(all_date_obses).strftime("%Y-%m-%dT%H:%M:%S")
    meta["DATE-OBS"] = reference_time.strftime("%Y-%m-%dT%H:%M:%S") if reference_time else meta["DATE-AVG"].value
    meta["DATE-BEG"] = min(all_date_obses).strftime("%Y-%m-%dT%H:%M:%S")
    meta["DATE-END"] = max(all_date_obses).strftime("%Y-%m-%dT%H:%M:%S")
    meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    meta.history.add_now("stray light",
                         f"Generated with {len(filepaths)} files running from "
                         f"{min(all_date_obses).strftime('%Y-%m-%dT%H:%M:%S')} to "
                         f"{max(all_date_obses).strftime('%Y-%m-%dT%H:%M:%S')}")
    meta["FILEVRSN"] = cube.meta["FILEVRSN"].value

    # Let's put in a valid, representative WCS, with the right scale and sun-relative pointing, etc.
    wcs = cube.wcs
    out_cube = NDCube(data=np.array(models), meta=meta, wcs=wcs, uncertainty=StdDevUncertainty(uncertainty))

    return [out_cube]


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
            msg=f"Incorrect TELESCOP value within {model.meta['FILENAME'].value}"
            raise IncorrectTelescopeError(msg)
        if model.meta["OBSLAYR1"].value != data_object.meta["OBSLAYR1"].value:
            msg=f"Incorrect polarization state within {model.meta['FILENAME'].value}"
            raise IncorrectPolarizationStateError(msg)
        if model.data.shape[1:] != data_object.data.shape:
            msg = f"Incorrect stray light function shape within {model.meta['FILENAME'].value}"
            raise InvalidDataError(msg)

    # Duplicate bin at top
    bin_centers = np.linspace(-180, 180, stray_light_before_model.shape[0] + 1)
    bin_width = 360 / stray_light_before_model.shape[0]
    crota = data_object.meta["CROTA"].value
    # CROTA falls within [-180, 180]

    for before_bin, after_bin in pairwise(range(len(bin_centers))):
        if bin_centers[before_bin] < crota <= bin_centers[after_bin]:
            break

    fpos = (crota - bin_centers[before_bin]) / bin_width
    if after_bin == len(bin_centers) - 1:
        after_bin = 0

    before_at_orbit_pos = (stray_light_before_model.data[before_bin] * (1 - fpos)
                           + stray_light_before_model.data[after_bin] * fpos)
    after_at_orbit_pos = (stray_light_after_model.data[before_bin] * (1 - fpos)
                          + stray_light_after_model.data[after_bin] * fpos)
    stray_light_before_model = NDCube(
            data=before_at_orbit_pos, meta=stray_light_before_model.meta, wcs=stray_light_before_model.wcs)
    stray_light_after_model = NDCube(
            data=after_at_orbit_pos, meta=stray_light_after_model.meta, wcs=stray_light_after_model.wcs)

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
