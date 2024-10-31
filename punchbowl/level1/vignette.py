import os
import pathlib
import warnings

from ndcube import NDCube
from prefect import get_run_logger, task

from punchbowl.data import load_ndcube_from_fits
from punchbowl.exceptions import (
    IncorrectPolarizationStateWarning,
    IncorrectTelescopeWarning,
    InvalidDataError,
    LargeTimeDeltaWarning,
    NoCalibrationDataWarning,
)


@task
def correct_vignetting_task(data_object: NDCube, vignetting_path: pathlib) -> NDCube:
    """
    Prefect task to correct the vignetting of an image.

    Vignetting is a reduction of an image's brightness or saturation toward the
    periphery compared to the image center, created by the optical path. The
    Vignetting Module will transform the data through a flat-field correction
    map, to cancel out the effects of optical vignetting created by distortions
    in the optical path. This module also corrects detector gain variation and
    offset.

    Correction maps will be 2048*2048 arrays, to match the input data, and
    built using the starfield brightness pattern. Mathematical Operation:

        I'_{i,j} = I_i,j * FF_{i,j}

    Where I_{i,j} is the number of counts in pixel i, j. I'_{i,j} refers to the
    modified value. FF_{i,j} is the small-scale flat field factor for pixel i,
    j. The correction mapping will take into account the orientation of the
    spacecraft and its position in the orbit.

    Uncertainty across the image plane is calculated using the modelled
    flat-field correction with stim lamp calibration data. Deviations from the
    known flat-field are used to calculate the uncertainty in a given pixel.
    The uncertainty is convolved with the input uncertainty layer to produce
    the output uncertainty layer.


    Parameters
    ----------
    data_object : PUNCHData
        data on which to operate

    vignetting_path : pathlib
        path to vignetting function to apply to input data

    Returns
    -------
    PUNCHData
        modified version of the input with the vignetting corrected

    """
    logger = get_run_logger()
    logger.info("correct_vignetting started")

    if vignetting_path is None:
        data_object.meta.history.add_now("LEVEL1-correct_vignetting", "Vignetting skipped")
        msg=f"Calibration file {vignetting_path} is unavailable, vignetting correction not applied"
        warnings.warn(msg, NoCalibrationDataWarning)
    elif not vignetting_path.exists():
        msg = f"File {vignetting_path} does not exist."
        raise InvalidDataError(msg)
    else:
        vignetting_function = load_ndcube_from_fits(vignetting_path)
        vignetting_function_date = vignetting_function.meta.astropy_time
        observation_date = data_object.meta.astropy_time
        if abs((vignetting_function_date - observation_date).to("day").value) > 14:
            msg = f"Calibration file {vignetting_path} contains data created greater than 2 weeks from the obsveration"
            warnings.warn(msg, LargeTimeDeltaWarning)
        if vignetting_function.meta["TELESCOP"].value != data_object.meta["TELESCOP"].value:
            msg = f"Incorrect TELESCOP value within {vignetting_path}"
            warnings.warn(msg, IncorrectTelescopeWarning)
        if vignetting_function.meta["OBSLAYR1"].value != data_object.meta["OBSLAYR1"].value:
            msg = f"Incorrect polarization state within {vignetting_path}"
            warnings.warn(msg, IncorrectPolarizationStateWarning)
        if vignetting_function.data.shape != data_object.data.shape:
            msg = f"Incorrect vignetting function shape within {vignetting_path}"
            raise InvalidDataError(msg)

        data_object.data[:, :] = data_object.data[:, :] / vignetting_function.data[:, :]
        data_object.uncertainty.array[:, :] /= vignetting_function.data[:, :]
        data_object.meta.history.add_now("LEVEL1-correct_vignetting",
                                         f"Vignetting corrected using {os.path.basename(str(vignetting_path))}")
        data_object.meta["CALVI"] = os.path.basename(str(vignetting_path))
    logger.info("correct_vignetting finished")
    return data_object
