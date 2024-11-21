from datetime import datetime

import numpy as np
from ndcube import NDCube
from numpy.polynomial import polynomial
from prefect import flow, get_run_logger
from quadprog import solve_qp
from scipy.interpolate import griddata

from punchbowl.data import NormalizedMetadata, load_ndcube_from_fits
from punchbowl.data.wcs import load_trefoil_wcs
from punchbowl.exceptions import InvalidDataError
from punchbowl.prefect import punch_task


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

    solution = np.zeros((input_vals.shape[1], cube.shape[1], cube.shape[2]))
    num_inputs = np.zeros((cube.shape[1], cube.shape[2]))
    for i in range(cube.shape[1]):
        for j in range(cube.shape[2]):
            time_series = cube[:, i, j]
            is_good = np.isfinite(time_series)
            time_series = time_series[is_good]
            c_iter = c[:, is_good]
            g_iter = np.matmul(c_iter, c_iter.T)
            num_inputs[i, j] = np.sum(is_good)
            if time_series.size < n_nonnan_required:
                this_solution = np.zeros(input_vals.shape[1])
            else:
                a = np.matmul(c_iter, time_series)
                try:
                    this_solution = solve_qp(g_iter, a, c_iter, time_series)[0]
                except ValueError:
                    this_solution = np.zeros(input_vals.shape[1])
            solution[:, i, j] = this_solution

    return np.asarray(solution), num_inputs

def model_fcorona_for_cube(xt: np.ndarray,
                          cube: np.ndarray,
                          min_brightness: float = 1E-16,
                          smooth_level: float | None = 1,
                          return_full_curves: bool=False,
                          ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Model the F corona given a list of times and a corresponding data cube, .

    Parameters
    ----------
    xt : np.ndarray
        time array
    cube : np.ndarray
        observation array
    min_brightness: float
        pixels dimmer than this value are set to nan and considered empty
    smooth_level : float | None
        If None, no smoothing is applied.
        Otherwise, the top and bottom `smooth_level` standard deviations of data are rejected.
    return_full_curves: bool
        If True, this function returns the full curve fitted to the time series at each pixel
        and the smoothed data cube. If False (default), only the curve's value at the central
        frame is returned, producing a model at one instant in time.

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
    cube[cube < min_brightness] = np.nan
    if smooth_level is not None:
        average = np.nanmean(cube, axis=0)
        std = np.nanstd(cube, axis=0)
        a, b, c = np.where(cube[:, ...] > (average + (smooth_level * std)))
        cube[a, b, c] = average[b, c]

        a, b, c = np.where(cube[:, ...] < (average - (smooth_level * std)))
        cube[a, b, c] = average[b, c]

    xt = np.array(xt)
    xt -= xt[0]

    input_array = np.c_[np.power(xt, 3), np.square(xt), xt, np.ones(len(xt))]
    coefficients, counts = solve_qp_cube(input_array, -cube)
    coefficients *= -1
    if return_full_curves:
        return polynomial.polyval(xt, coefficients[::-1, :, :]).transpose((2, 0, 1)), counts, cube
    return polynomial.polyval(xt[len(xt)//2], coefficients[::-1, :, :]), counts


@flow
def construct_f_corona_background(
    data_list: list[str],
    layer: int,
    smooth_level: float = 1.0,
) -> tuple[NDCube, np.ndarray]:
    """Build f corona background model."""
    logger = get_run_logger()
    logger.info("construct_f_corona_background started")

    if len(data_list) == 0:
        msg = "data_list cannot be empty"
        raise ValueError(msg)

    data_list.sort()

    output = load_ndcube_from_fits(data_list[0])
    output_wcs = output.wcs
    output_meta = output.meta
    output_mask = output.mask

    data_shape = output.data[layer].shape

    number_of_data_frames = len(data_list)
    data_cube = np.empty((number_of_data_frames, *data_shape), dtype=float)
    uncertainty_cube = np.empty((number_of_data_frames, *data_shape), dtype=float)
    meta_list = []
    obs_times = []

    logger.info("beginning data loading")
    for i, address_out in enumerate(data_list):
        data_object = load_ndcube_from_fits(address_out)
        data_cube[i, :, :] = data_object.data[layer]
        uncertainty_cube[i, :, :] = data_object.uncertainty.array[layer]
        obs_times.append(data_object.meta.datetime.timestamp())
        meta_list.append(data_object.meta)
    logger.info("ending data loading")

    logger.info("building model")
    f_background, counts = model_fcorona_for_cube(obs_times, data_cube, smooth_level=smooth_level)
    output = NDCube(f_background, wcs=output_wcs, meta=output_meta, mask=output_mask)

    logger.info("construct_f_corona_background finished")
    output.meta.history.add_now("LEVEL3-construct_f_corona_background", "constructed f corona model")

    return output, counts

def fill_nans_with_interpolation(image: np.ndarray) -> np.ndarray:
    """Fill NaN values in an image using interpolation."""
    mask = np.isnan(image)
    x, y = np.where(~mask)
    known_values = image[~mask]

    grid_x, grid_y = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    return griddata((x, y), known_values, (grid_x, grid_y), method="cubic")

@flow(log_prints=True)
def construct_full_f_corona_model(filenames: list[str]) -> list[NDCube]:
    """Construct a full F corona model."""
    trefoil_wcs, trefoil_shape = load_trefoil_wcs()

    m_model_fcorona, _ = construct_f_corona_background(filenames, 0, smooth_level=3.0)
    m_model_fcorona.data[m_model_fcorona.data==0] = np.nan
    m_model_fcorona = fill_nans_with_interpolation(m_model_fcorona.data)

    z_model_fcorona, _ = construct_f_corona_background(filenames, 1, smooth_level=3.0)
    z_model_fcorona.data[z_model_fcorona.data==0] = np.nan
    z_model_fcorona = fill_nans_with_interpolation(z_model_fcorona.data)

    p_model_fcorona, _ = construct_f_corona_background(filenames, 2, smooth_level=3.0)
    p_model_fcorona.data[p_model_fcorona.data==0] = np.nan
    p_model_fcorona = fill_nans_with_interpolation(p_model_fcorona.data)

    meta = NormalizedMetadata.load_template("PFM", "3")
    meta["DATE-OBS"] = str(datetime(2024, 8, 1, 12, 0, 0,
                                    tzinfo=datetime.timezone.utc))
    before_cube = NDCube(data=np.stack([m_model_fcorona,
                                               z_model_fcorona,
                                               p_model_fcorona], axis=0),
                                meta=meta,
                                wcs=trefoil_wcs)

    meta = NormalizedMetadata.load_template("PFM", "3")
    meta["DATE-OBS"] =  str(datetime(2024, 12, 1, 12, 0, 0,
                                     tzinfo=datetime.timezone.utc))
    after_cube = NDCube(data=np.stack([m_model_fcorona,
                                               z_model_fcorona,
                                               p_model_fcorona], axis=0),
                                meta=meta,
                                wcs=trefoil_wcs)

    return [before_cube, after_cube]

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

    before_date = before_f_background_model.meta.datetime.timestamp()
    after_date = after_f_background_model.meta.datetime.timestamp()
    observation_date = data_object.meta.datetime.timestamp()

    if before_date > observation_date:
        msg = "Before F corona model was after the observation date"
        raise InvalidDataError(msg)

    if after_date < observation_date:
        msg = "After F corona model was before the observation date"
        raise InvalidDataError(msg)

    if before_date == observation_date:
        interpolated_model = before_f_background_model
    elif after_date == observation_date:
        interpolated_model = after_f_background_model
    else:
        interpolated_model = ((after_f_background_model.data - before_f_background_model.data)
                              * (observation_date - before_date) / (after_date - before_date)
                              + before_f_background_model.data)

    interpolated_model[np.isinf(data_object.uncertainty.array)] = 0

    data_object.data[...] = data_object.data[...] - interpolated_model
    data_object.uncertainty.array[:, :] -= interpolated_model
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
