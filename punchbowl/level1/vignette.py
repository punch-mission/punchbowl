import pathlib
import warnings

from ndcube import NDCube
from prefect import get_run_logger, task

from punchbowl.data import load_ndcube_from_fits
from punchbowl.exceptions import InvalidDataError


@task
def correct_vignetting_task(data_object: NDCube, vignetting_file: pathlib) -> NDCube:
    """
    Prefect task to correct the vignetting of an image.

    Parameters
    ----------
    data_object : PUNCHData
        data on which to operate
    vignetting_file: pathlib
        path to vignetting function to apply to input data

    Returns
    -------
    PUNCHData
        modified version of the input with the vignetting corrected

    """
    logger = get_run_logger()
    logger.info("correct_vignetting started")

    if vignetting_file is None:
        data_object.meta.history.add_now("LEVEL1-correct_vignetting", "Vignetting skipped")
    elif not vignetting_file.exists():
        msg = f"File {vignetting_file} does not exist."
        raise InvalidDataError(msg)
    else:
        vignetting_function = load_ndcube_from_fits(vignetting_file)

        if vignetting_function.meta["TELESCOP"].value != data_object.meta["TELESCOP"].value:
            warnings.warn(f"Incorrect TELESCOP value within {vignetting_file}", UserWarning)
        elif vignetting_function.meta["OBSLAYR1"].value != data_object.meta["OBSLAYR1"].value:
            warnings.warn(f"Incorrect polarization state within {vignetting_file}", UserWarning)
        elif vignetting_function.data.shape != data_object.data.shape:
            msg = f"Incorrect vignetting function shape within {vignetting_file}"
            raise InvalidDataError(msg)
        else:
            data_object.data[:, :] /= vignetting_function.data[:, :]
            data_object.meta.history.add_now("LEVEL1-correct_vignetting", "Vignetting corrected")

    logger.info("correct_vignetting finished")
    return data_object
