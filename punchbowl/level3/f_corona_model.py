import numpy as np
from ndcube import NDCube
from numpy.polynomial import polynomial
from prefect import get_run_logger, task
from quadprog import solve_qp

from punchbowl.data import load_ndcube_from_fits
from punchbowl.exceptions import InvalidDataError


def solve_qp_cube(input_vals: np.ndarray, cube: np.ndarray) -> np.ndarray:
    """
    Fast solver for the quadratic programming problem.

    Parameters
    ----------
    input_vals : np.ndarray
        array of times
    cube : np.ndarray
        array of data

    Returns
    -------
    np.ndarray
        Array of coefficients for solving polynomial

    """
    c = np.transpose(input_vals)
    g = np.matmul(c, input_vals)

    sol = np.zeros((input_vals.shape[1], cube.shape[1], cube.shape[2]))

    for i in range(cube.shape[1]):
        for j in range(cube.shape[2]):
            a = np.matmul(c, cube[:, i, j])
            try:
                this_sol = solve_qp(g, a, c, cube[:, i, j])
            except ValueError:
                this_sol = np.zeros(input_vals.shape[1])
            for k in range(sol.shape[0]):
                sol[k, i, j] = this_sol[k]

    return np.asarray(sol)

def model_fcorona_for_cube(xt: np.ndarray,
                          cube: np.ndarray,
                          smooth_level: float | None =4) -> np.ndarray:
    """
    Model the F corona given a list of times and a corresponding data cube, .

    Parameters
    ----------
    xt : np.ndarray
        time array
    cube : np.ndarray
        observation array
    smooth_level : float | None
        If None, no smoothing is applied.
        Otherwise, the top and bottom `smooth_level` standard deviations of data are rejected.

    Returns
    -------
    np.ndarray

    """
    average = np.mean(cube, axis=0)
    std = np.nanstd(cube, axis=0)
    if smooth_level is not None:
        a, b, c = np.where(cube[:, ...] > (average - (smooth_level * std)))
        cube[a, b, c] = average[b, c]

        a, b, c = np.where(cube[:, ...] < (average - (smooth_level * std)))
        cube[a, b, c] = average[b, c]

    input_array = np.c_[np.power(xt, 3), np.square(xt), xt, np.ones(len(xt))]
    out = -solve_qp_cube(input_array, -cube)
    return polynomial.polyval(xt[len(xt)//2], out[::-1, :, :])

@task
def construct_f_corona_background(
    data_list: list[str],
    layer: int,
) -> NDCube:
    """Build f corona background model."""
    logger = get_run_logger()
    logger.info("construct_f_corona_background started")

    if len(data_list) == 0:
        msg = "data_list cannot be empty"
        raise ValueError(msg)

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

    for i, address_out in enumerate(data_list):
        data_object = load_ndcube_from_fits(address_out)
        data_cube[i, :, :] = data_object.data[layer]
        uncertainty_cube[i, :, :] = data_object.uncertainty.array[layer]
        obs_times.append(data_object.meta.datetime.timestamp())
        meta_list.append(data_object.meta)

    f_background = model_fcorona_for_cube(obs_times, data_cube)
    output = NDCube(f_background, wcs=output_wcs, meta=output_meta, mask=output_mask)

    logger.info("construct_f_corona_background finished")
    output.meta.history.add_now("LEVEL3-construct_f_corona_background", "constructed f corona model")

    return output


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

@task
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
