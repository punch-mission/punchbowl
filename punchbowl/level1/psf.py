import os
from pathlib import Path

from ndcube import NDCube
from prefect import get_run_logger, task
from regularizepsf.transform import ArrayPSFTransform


def correct_psf(
    data: NDCube,
    psf_transform: ArrayPSFTransform,
) -> NDCube:
    """Correct PSF."""
    new_data = psf_transform.apply(data.data)

    data.data[...] = new_data[...]
    # TODO: uncertainty propagation
    return data

@task
def correct_psf_task(
    data_object: NDCube,
    model_path: str | None = None,
) -> NDCube:
    """
    Prefect Task to correct the PSF of an image.

    Parameters
    ----------
    data_object : PUNCHData
        data to operate on
    model_path : str
        path to the PSF model to use in the correction

    Returns
    -------
    PUNCHData
        modified version of the input with the PSF corrected

    """
    logger = get_run_logger()
    logger.info("correct_psf started")

    if model_path is not None:
        corrector = ArrayPSFTransform.load(Path(model_path))
        data_object = correct_psf(data_object, corrector)
        data_object.meta.history.add_now("LEVEL1-correct_psf",
                                         f"PSF corrected with {os.path.basename(model_path)} model")
        data_object.meta["CALPSF"] = os.path.basename(model_path)
    else:
        data_object.meta.history.add_now("LEVEL1-correct_psf", "Empty model path so no correction applied")
        logger.info("No model path so PSF correction is skipped")

    logger.info("correct_psf finished")
    return data_object
