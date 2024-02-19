import pathlib

from astropy.io import fits
from prefect import get_run_logger, task

from punchbowl.data import PUNCHData


@task
def correct_vignetting_task(data_object: PUNCHData, vignetting_file: pathlib) -> PUNCHData:
    """Prefect task to correct the vignetting of an image

    Parameters
    ----------
    data_object : PUNCHData
        data to operate on
    vignetting_file: pathlib
        path to vignetting function to apply to input data

    Returns
    -------
    PUNCHData
        modified version of the input with the vignetting corrected
    """
    logger = get_run_logger()
    logger.info("correct_vignetting started")

    # TODO - Check for correct vignetting file (check spacecraft, etc?)
    # TODO - Handle incorrect size array?
    # TODO - Sort out vignetting function slice to take

    if vignetting_file is not None:
        if not vignetting_file.exists():
            raise OSError(f"File {vignetting_file} does not exist.")
        else:
            with fits.open(vignetting_file) as hdul:
                vignetting_function = hdul[1].data

            data_object.data[:,:] /= vignetting_function[:,:]
            data_object.meta.history.add_now("LEVEL1-correct_vignetting", "Vignetting corrected")
    else:
        data_object.meta.history.add_now("LEVEL1-correct_vignetting", "Vignetting skipped")

    logger.info("correct_vignetting finished")
    return data_object
