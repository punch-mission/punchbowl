import os
import pathlib
import warnings
from collections.abc import Callable

from ndcube import NDCube

from punchbowl.data import load_ndcube_from_fits
from punchbowl.exceptions import (
    IncorrectPolarizationStateWarning,
    IncorrectTelescopeWarning,
    InvalidDataError,
    LargeTimeDeltaWarning,
    NoCalibrationDataWarning,
)
from punchbowl.prefect import punch_task


@punch_task
def dark_subtraction_task(data_object: NDCube, dark_path: str | pathlib.Path | Callable | None) -> NDCube:
    """
    Prefect task to perform dark subtraction on input data.

    Images are taken with no illumination on the CCD to characterize the dark noise
    distribution. These calibrations can then be subtracted from subsequent data
    to remove these contributions.

    Correction maps will be 2048*2048 arrays, to match the input data.
    Mathematical Operation:

        I'_{i,j} = I_i,j - DK_{i,j}

    Where I_{i,j} is the number of counts in pixel i, j. I'_{i,j} refers to the
    modified value. DK_{i,j} is the dark image value for pixel i, j.

    Parameters
    ----------
    data_object : NDCube
        data on which to operate

    dark_path : pathlib
        path to dark image to apply to input data

    Returns
    -------
    NDCube
        modified version of the input with the dark subtracted

    """
    if dark_path is None:
        data_object.meta.history.add_now("LEVEL1-dark_subtraction", "Dark subtraction skipped")
        msg=f"Calibration file {dark_path} is unavailable, dark correction not applied"
        warnings.warn(msg, NoCalibrationDataWarning)
    else:
        if isinstance(dark_path, Callable):
            dark_frame, dark_path = dark_path()
        else:
            if isinstance(dark_path, str):
                dark_path = pathlib.Path(dark_path)
            if not dark_path.exists():
                msg = f"File {dark_path} does not exist."
                raise InvalidDataError(msg)
            dark_frame = load_ndcube_from_fits(dark_path, include_provenance=False)
        dark_frame_date = dark_frame.meta.astropy_time
        observation_date = data_object.meta.astropy_time
        # TODO - How long is a dark frame good for?
        if abs((dark_frame_date - observation_date).to("day").value) > 14:
            msg = f"Calibration file {dark_path} contains data created greater than 2 weeks from the observation"
            warnings.warn(msg, LargeTimeDeltaWarning)
        if dark_frame.meta["TELESCOP"].value != data_object.meta["TELESCOP"].value:
            msg = f"Incorrect TELESCOP value within {dark_path}"
            warnings.warn(msg, IncorrectTelescopeWarning)
        if dark_frame.meta["OBSLAYR1"].value != data_object.meta["OBSLAYR1"].value:
            msg = f"Incorrect polarization state within {dark_path}"
            warnings.warn(msg, IncorrectPolarizationStateWarning)
        if dark_frame.data.shape != data_object.data.shape:
            msg = f"Incorrect dark function shape within {dark_path}"
            raise InvalidDataError(msg)

        data_object.data[:, :] -= dark_frame.data[:, :]
        # TODO - need to properly propagate uncertainty using the uncertainty propagation function
        data_object.uncertainty.array[:,:] -= dark_frame.uncertainty.array[:,:]
        data_object.meta.history.add_now("LEVEL1-dark_subtraction",
                                         f"Dark subtracted using {os.path.basename(str(dark_path))}")
        data_object.meta["CALDK"] = os.path.basename(str(dark_path))
    return data_object
