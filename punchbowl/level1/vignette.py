import pathlib
import warnings

import numpy as np
from astropy.nddata import StdDevUncertainty
from prefect import get_run_logger, task

from punchbowl.data import PUNCHData
from punchbowl.exceptions import InvalidDataError


@task
def correct_vignetting_task(data_object: PUNCHData, vignetting_file: pathlib) -> PUNCHData:
    """Prefect task to correct the vignetting of an image

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
        raise InvalidDataError(f"File {vignetting_file} does not exist.")
    else:
        vignetting_function = PUNCHData.from_fits(vignetting_file)

        if vignetting_function.meta["TELESCOP"].value != data_object.meta["TELESCOP"].value:
            warnings.warn(f"Incorrect TELESCOP value within {vignetting_file}", UserWarning)
        elif vignetting_function.meta["OBSLAYR1"].value != data_object.meta["OBSLAYR1"].value:
            warnings.warn(f"Incorrect polarization state within {vignetting_file}", UserWarning)
        elif vignetting_function.data.shape != data_object.data.shape:
            raise InvalidDataError(f"Incorrect vignetting function shape within {vignetting_file}")
        else:
            new_data = data_object.data[:, :] / vignetting_function.data[:, :]

            new_uncertainty = StdDevUncertainty(np.maximum(new_data, data_object.data) /
                                                np.minimum(new_data, data_object.data) * data_object.uncertainty.array)

            data_object = data_object.duplicate_with_updates(data=new_data, uncertainty=new_uncertainty)

            data_object.meta.history.add_now("LEVEL1-correct_vignetting", "Vignetting corrected")

    logger.info("correct_vignetting finished")
    return data_object
