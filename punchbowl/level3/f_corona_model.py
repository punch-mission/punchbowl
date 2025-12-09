import multiprocessing as mp
from datetime import UTC, datetime

import astropy
import numpy as np
import scipy.optimize
from astropy.nddata import StdDevUncertainty
from dateutil.parser import parse as parse_datetime_str
from ndcube import NDCube
from numpy.polynomial import polynomial
from prefect import get_run_logger
from quadprog import solve_qp
from scipy.interpolate import griddata
from threadpoolctl import threadpool_limits

from punchbowl.data import NormalizedMetadata, load_ndcube_from_fits
from punchbowl.data.punch_io import load_many_cubes_iterable
from punchbowl.data.wcs import load_trefoil_wcs
from punchbowl.exceptions import InvalidDataError
from punchbowl.prefect import punch_flow, punch_task
from punchbowl.util import interpolate_data, masked_mean, nan_percentile


def solve_qp_cube(input_vals: np.ndarray, cube: np.ndarray,
                  n_nonnan_required: int=7) -> (np.ndarray, np.ndarray):
    """
    Fast solver for the quadratic programming problem.

    Parameters
    ----------
    input_vals : np.ndarray
        array of times
    cube : np.ndarray
        array of data
    n_nonnan_required : int
        The number of non-nan values that must be present in each pixel's time series.
        Any pixels with fewer will not be fit, with zeros returned instead.

    Returns
    -------
    np.ndarray
        Array of coefficients for solving polynomial

    """
    c = np.transpose(input_vals)
    cube_is_good = np.isfinite(cube)
    num_inputs = np.sum(cube_is_good, axis=0)

    solution = np.zeros((input_vals.shape[1], cube.shape[1], cube.shape[2]))
    for i in range(cube.shape[1]):
        for j in range(cube.shape[2]):
            is_good = cube_is_good[:, i, j]
            time_series = cube[:, i, j][is_good]
            if time_series.size < n_nonnan_required:
                solution[:, i, j] = 0
            else:
                c_iter = c[:, is_good]
                g_iter = np.matmul(c_iter, c_iter.T)
                a = np.matmul(c_iter, time_series)
                try:
                    solution[:, i, j] = solve_qp(g_iter, a, c_iter, time_series)[0]
                except ValueError:
                    solution[:, i, j] = 0
    return np.asarray(solution), num_inputs


def model_fcorona_for_cube_real(xt: np.ndarray,
                           reference_xt: float,
                           cube: np.ndarray,
                           min_brightness: float = 1E-18,
                           clip_factor: float | None = 1,
                           return_full_curves: bool = False,
                           num_workers: int | None = 8,
                           detrend: bool = True,
                           ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Model the F corona given a list of times and a corresponding data cube.

    Parameters
    ----------
    xt : np.ndarray
        time array
    reference_xt: float
        timestamp to evaluate the model for
    cube : np.ndarray
        observation array
    min_brightness: float
        pixels dimmer than this value are set to nan and considered empty
    clip_factor : float | None
        If None, no smoothing is applied.
        Otherwise, the difference between the 25th and 75th percentile is computed and values that vary from the median
        by more than `clip_factor` times the difference data are rejected.
    return_full_curves: bool
        If True, this function returns the full curve fitted to the time series at each pixel
        and the smoothed data cube. If False (default), only the curve's value at the central
        frame is returned, producing a model at one instant in time.
    num_workers: int | None
        Work is parallelized over this many worker processes. If None, this matches the number of cores.
    detrend : bool
        Whether to detrend each time series before outlier rejection

    Returns
    -------
    np.ndarray
        The F-corona model at the central point in time. If return_full_curves is True, this is
        instead the F-corona model at all points in time covered by the data cube
    np.ndarray
        The number of data points used in solving the F-corona model for each pixel of the output
    np.ndarray
        The smoothed data cube. Returned only if return_full_curves is True.

    """
    # TODO : re-enable F corona modeling
    stride = 32
    def args() -> tuple:
        # Generate a set of args for one task
        for i in range(0, cube.shape[0], stride):
            for j in range(0, cube.shape[1], stride):
                yield (xt, reference_xt, cube[i:i+stride, j:j+stride, :], min_brightness, clip_factor,
                       return_full_curves, detrend)

    def reassemble(inputs: tuple) -> np.ndarray:
        output = np.empty((cube.shape[0], cube.shape[1], *inputs[0].shape[2:]), dtype=inputs[0].dtype)
        k = 0
        for i in range(0, cube.shape[0], stride):
            for j in range(0, cube.shape[1], stride):
                output[i:i+stride, j:j+stride] = inputs[k]
                k += 1
        return output


    # Since we're parallelizing with processes, we shouldn't run a lot of threads
    with threadpool_limits(2), mp.Pool(processes=num_workers) as pool:
        chunks = pool.starmap(_model_fcorona_for_cube_inner, args(), chunksize=4)

    # Combine the outputs of each task into final output arrays
    if return_full_curves:
        curves, counts, cubes = zip(*chunks, strict=False)
        curves = reassemble(curves)
        counts = reassemble(counts)
        cubes = reassemble(cubes)
        return curves, counts, cubes
    model, counts = zip(*chunks, strict=False)
    model = reassemble(model)
    counts = reassemble(counts)

    return model, counts


def model_fcorona_for_cube(xt: np.ndarray, # noqa: ARG001
                           reference_xt: float, # noqa: ARG001
                           cube: np.ndarray,
                           *args: list, **kwargs: dict, # noqa: ARG001
                           ) -> tuple[np.ndarray, np.ndarray]:
    """
    Model the F corona given a list of times and a corresponding data cube.

    Parameters
    ----------
    xt : np.ndarray
        Unused
    reference_xt: float
        Unused
    cube : np.ndarray
        observation array
    args : list
        Kept for signature compatibility
    kwargs : dict
        Kept for signature compatibility

    Returns
    -------
    np.ndarray
        The F-corona model at the central point in time. If return_full_curves is True, this is
        instead the F-corona model at all points in time covered by the data cube
    None
        Nothing

    """
    cube[cube == 0] = np.nan
    return nan_percentile(cube, 3), None


def model_polarized_fcorona_for_cube(xt: np.ndarray, # noqa: ARG001
                                     reference_xt: float, # noqa: ARG001
                                     cube: np.ndarray,
                                     low_percentile: float = 5.0,
                                     high_percentile: float = 10.0,
                                     *args: list, **kwargs: dict, # noqa: ARG001
                                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the polarized f corona model using indexing method."""
    cube[cube == 0] = np.nan

    mdata = cube[:, 0, :, :]
    zdata = cube[:, 1, :, :]
    pdata = cube[:, 2, :, :]
    # Estimate total brightness
    tbcube = 2 / 3 * np.sum(cube, axis=1)

    # Per-pixel percentile threshold of tbcube over time (T axis)
    low_thresh, high_thresh = nan_percentile(tbcube, [low_percentile, high_percentile])
    mask = (tbcube <= high_thresh) * (tbcube >= low_thresh)  # shape: (T, H, W)

    # We don't need this anymore and we're holding a lot of RAM, so release some
    del tbcube

    # Estimate MZP background based on index
    m_background = masked_mean(mdata, mask)
    z_background = masked_mean(zdata, mask)
    p_background = masked_mean(pdata, mask)

    return m_background, z_background, p_background

def _model_fcorona_for_cube_inner(xt: np.ndarray,
                                  reference_xt: float,
                                  cube: np.ndarray,
                                  min_brightness: float = 1E-18,
                                  clip_factor: float | None = 1,
                                  return_full_curves: bool=False,
                                  detrend: bool = True,
                                  ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    cube = cube.transpose((2, 0, 1))
    cube[cube < min_brightness] = np.nan
    xt = np.array(xt)
    reference_xt -= xt[0]
    xt -= xt[0]

    def trend_fcn(x: np.ndarray, xvals: np.ndarray) -> np.ndarray:
        c0, c1, c2 = x
        return c0 + c1 * xvals + c2 * xvals ** 2

    def trend_resid(x: np.ndarray, xvals: np.ndarray, yvals: np.ndarray) -> np.ndarray:
        return trend_fcn(x, xvals.ravel()) - yvals.ravel()

    good_px = np.isfinite(cube)
    if detrend:
        if np.sum(good_px) < 20:
            detrended_cube = cube
        else:
            x = np.broadcast_to(xt[:, None, None], cube.shape)
            jacobian = np.stack((
                    0*x[good_px].ravel() + 1,
                    x[good_px],
                    x[good_px] ** 2,
                ), axis=1)
            res = scipy.optimize.least_squares(trend_resid, (np.median(cube[good_px]), 0, 0), loss="cauchy",
                                               f_scale=.5e-13, kwargs={"xvals": x[good_px], "yvals": cube[good_px]},
                                               jac=lambda *a, **kw: jacobian) #noqa: ARG005
            trend = trend_fcn(res.x, xt)
            detrended_cube = cube - trend[:, None, None]
    else:
        detrended_cube = cube

    if clip_factor is not None and np.any(good_px):
        low, center, high = nan_percentile(detrended_cube, [25, 50, 75])
        width = high - low
        a, b, c = np.where(detrended_cube[:, ...] > (center + (clip_factor * width)))
        cube[a, b, c] = np.nan

        a, b, c = np.where(detrended_cube[:, ...] < (center - (clip_factor * width)))
        cube[a, b, c] = np.nan

    input_array = np.c_[np.power(xt, 3), np.square(xt), xt, np.ones(len(xt))]
    coefficients, counts = solve_qp_cube(input_array, -cube)
    coefficients *= -1
    if return_full_curves:
        return polynomial.polyval(xt, coefficients[::-1, :, :]), counts, cube.transpose((1, 2, 0))

    return polynomial.polyval(reference_xt, coefficients[::-1, :, :]), counts


def fill_nans_with_interpolation(image: np.ndarray) -> np.ndarray:
    """Fill NaN values in an image using interpolation."""
    mask = np.isnan(image)
    x, y = np.where(~mask)
    known_values = image[~mask]

    grid_x, grid_y = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    return griddata((x, y), known_values, (grid_x, grid_y), method="cubic")


@punch_flow(log_prints=True)
def construct_f_corona_model(filenames: list[str], # noqa: C901
                             clip_factor: float = 3.0,
                             reference_time: str | None = None,
                             num_workers: int = 8,
                             num_loaders: int | None = None,
                             fill_nans: bool = False,
                             polarized: bool = False) -> list[NDCube]:
    """Construct a full F corona model."""
    logger = get_run_logger()

    if reference_time is None:
        reference_time = datetime.now(UTC)
    elif isinstance(reference_time, str):
        reference_time = parse_datetime_str(reference_time)

    trefoil_wcs, trefoil_shape = load_trefoil_wcs()

    logger.info("construct_f_corona_background started")

    if len(filenames) == 0:
        msg = "Require at least one input file"
        raise ValueError(msg)

    filenames.sort()

    data_shape = (3, *trefoil_shape) if polarized else trefoil_shape

    number_of_data_frames = len(filenames)
    data_cube = np.empty((number_of_data_frames, *data_shape), dtype=float)

    meta_list = []
    obs_times = []

    logger.info("beginning data loading")
    dates = []
    n_failed = 0
    j = 0
    for i, result in enumerate(load_many_cubes_iterable(filenames, allow_errors=True, n_workers=num_loaders)):
        if isinstance(result, str):
            logger.warning(f"Loading {filenames[i]} failed")
            logger.warning(result)
            n_failed += 1
            if n_failed > 10:
                raise RuntimeError(f"{n_failed} files failed to load, stopping")
            continue
        cube = result
        dates.append(cube.meta.datetime)
        data_cube[j] = np.where(np.isnan(cube.uncertainty.array), np.nan, cube.data)
        j += 1
        obs_times.append(cube.meta.datetime.timestamp())
        meta_list.append(cube.meta)
        if (i + 1) % 50 == 0:
            logger.info(f"Loaded {i+1}/{len(filenames)} files")
    # Crop the unused end of the array if we had a few files that errored out
    data_cube = data_cube[:j+1]
    logger.info("end of data loading")
    output_datebeg = min(dates).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    output_dateend = max(dates).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    reference_xt = reference_time.timestamp()
    if polarized:
        m_model_fcorona, z_model_fcorona, p_model_fcorona = model_polarized_fcorona_for_cube(
                                                    obs_times, reference_xt,
                                                    data_cube,
                                                    num_workers=num_workers,
                                                    percentile=clip_factor)

        m_model_fcorona[m_model_fcorona == 0] = np.nan
        z_model_fcorona[z_model_fcorona == 0] = np.nan
        p_model_fcorona[p_model_fcorona == 0] = np.nan
        if fill_nans:
            m_model_fcorona = fill_nans_with_interpolation(m_model_fcorona)
            z_model_fcorona = fill_nans_with_interpolation(z_model_fcorona)
            p_model_fcorona = fill_nans_with_interpolation(p_model_fcorona)

        output_data = np.stack([m_model_fcorona,
                                z_model_fcorona,
                                p_model_fcorona], axis=0)
        uncertainty = np.sqrt(np.abs(output_data)) / np.sqrt(len(obs_times))
        meta = NormalizedMetadata.load_template("PFM", "3")
        trefoil_wcs = astropy.wcs.utils.add_stokes_axis_to_wcs(trefoil_wcs, 2)
    else:
        model_fcorona, _ = model_fcorona_for_cube(obs_times, reference_xt,
                                                  data_cube,
                                                  num_workers=num_workers,
                                                  clip_factor=clip_factor)
        # model_fcorona[model_fcorona==0] = np.nan # noqa: ERA001
        if fill_nans:
            model_fcorona = fill_nans_with_interpolation(model_fcorona)

        output_data = model_fcorona
        uncertainty = np.sqrt(np.abs(model_fcorona)) / np.sqrt(len(obs_times))
        meta = NormalizedMetadata.load_template("CFM", "3")

    meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    meta["DATE-AVG"] = reference_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    meta["DATE-OBS"] = reference_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    meta["DATE-BEG"] = output_datebeg
    meta["DATE-END"] = output_dateend

    output_cube = NDCube(data=output_data,
                         meta=meta,
                         wcs=trefoil_wcs,
                         uncertainty=StdDevUncertainty(uncertainty))


    return [output_cube]

def subtract_f_corona_background(data_object: NDCube,
                                 before_f_background_model: NDCube,
                                 after_f_background_model: NDCube ) -> NDCube:
    """Subtract f corona background."""
    # check dimensions match
    if data_object.data.shape != before_f_background_model.data.shape:
        msg = (
            "f_background_subtraction expects the data_object and"
            "f_background arrays to have the same dimensions."
            f"data_array dims: {data_object.data.shape} "
            f"and before_f_background_model dims: {before_f_background_model.data.shape}"
        )
        raise InvalidDataError(
            msg,
        )

    if data_object.data.shape != after_f_background_model.data.shape:
        msg = (
            "f_background_subtraction expects the data_object and"
            "f_background arrays to have the same dimensions."
            f"data_array dims: {data_object.data.shape} "
            f"and after_f_background_model dims: {after_f_background_model.data.shape}"
        )
        raise InvalidDataError(
            msg,
        )

    interpolated_model, interpolated_uncertainty = interpolate_data(
            before_f_background_model,
            after_f_background_model,
            data_object.meta.datetime,
            and_uncertainty=True)

    interpolated_model[np.isinf(data_object.uncertainty.array)] = 0

    original_mask = (data_object.data == 0) * np.isinf(data_object.uncertainty.array)
    data_object.data[...] -= interpolated_model
    data_object.data[original_mask] = 0
    data_object.uncertainty.array[...] = np.sqrt(data_object.uncertainty.array**2 + interpolated_uncertainty**2)
    return data_object

@punch_task
def subtract_f_corona_background_task(observation: NDCube,
                                      before_f_background_model_path: str,
                                      after_f_background_model_path: str) -> NDCube:
    """
    Subtracts a background f corona model from an observation.

    This algorithm linearly interpolates between the before and after models.

    Parameters
    ----------
    observation : NDCube
        an observation to subtract an f corona model from

    before_f_background_model_path : str
        path to a NDCube f corona background map before the observation

    after_f_background_model_path : str
        path to a NDCube f corona background map after the observation

    Returns
    -------
    NDCube
        A background subtracted data frame

    """
    logger = get_run_logger()
    logger.info("subtract_f_corona_background started")


    before_f_corona_model = load_ndcube_from_fits(before_f_background_model_path)
    after_f_corona_model = load_ndcube_from_fits(after_f_background_model_path)

    output = subtract_f_corona_background(observation, before_f_corona_model, after_f_corona_model)
    output.meta.history.add_now("LEVEL3-subtract_f_corona_background", "subtracted f corona background")

    logger.info("subtract_f_corona_background finished")

    return output


def create_empty_f_background_model(data_object: NDCube) -> np.ndarray:
    """Create an empty background model."""
    return np.zeros_like(data_object.data)
