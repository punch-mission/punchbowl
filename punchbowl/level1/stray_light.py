import pathlib
import warnings

from astropy.nddata import StdDevUncertainty
from prefect import get_run_logger, task

from punchbowl.data import PUNCHData
from punchbowl.exceptions import InvalidDataError


@task
def remove_stray_light_task(data_object: PUNCHData, stray_light_path: pathlib) -> PUNCHData:
    """Prefect task to remove stray light from an image

    Parameters
    ----------
    data_object : PUNCHData
        data to operate on
    stray_light_path: pathlib
        path to stray light model to apply to data

    Returns
    -------
    PUNCHData
        modified version of the input with the stray light removed
    """
    logger = get_run_logger()
    logger.info("remove_stray_light started")

    if stray_light_path is None:
        data_object.meta.history.add_now("LEVEL1-remove_stray_light", "Stray light correction skipped")
    elif not stray_light_path.exists():
        raise InvalidDataError(f"File {stray_light_path} does not exist.")
    else:
        stray_light_model = PUNCHData.from_fits(stray_light_path)

        if stray_light_model.meta["TELESCOP"].value != data_object.meta["TELESCOP"].value:
            warnings.warn(f"Incorrect TELESCOP value within {stray_light_path}", UserWarning)
        elif stray_light_model.meta["OBSLAYR1"].value != data_object.meta["OBSLAYR1"].value:
            warnings.warn(f"Incorrect polarization state within {stray_light_path}", UserWarning)
        elif stray_light_model.data.shape != data_object.data.shape:
            raise InvalidDataError(f"Incorrect vignetting function shape within {stray_light_path}")
        else:
            new_data = data_object.data[:, :] - stray_light_model.data[:, :]
            new_uncertainty = StdDevUncertainty(data_object.uncertainty.array[:, :] - stray_light_model.data[:, :])

            data_object = data_object.duplicate_with_updates(data=new_data, uncertainty=new_uncertainty)

            data_object.meta.history.add_now("LEVEL1-remove_stray_light", "stray light removed")

    logger.info("remove_stray_light finished")
    return data_object
