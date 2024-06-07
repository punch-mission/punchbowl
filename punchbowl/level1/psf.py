import typing as t

from prefect import get_run_logger, task
from regularizepsf.corrector import ArrayCorrector

from punchbowl.data import PUNCHData


def correct_psf(
    data: PUNCHData,
    corrector: ArrayCorrector,
    alpha: float = 0,
    epsilon: float = 0.035,
) -> PUNCHData:
    new_data = corrector.correct_image(data.data, alpha=alpha, epsilon=epsilon)

    return data.duplicate_with_updates(data=new_data)


@task
def correct_psf_task(
    data_object: PUNCHData,
    model_path: t.Optional[str] = None,
) -> PUNCHData:
    """Prefect Task to correct the PSF of an image

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
        corrector = ArrayCorrector.load(model_path)
        data_object = correct_psf(data_object, corrector)
        data_object.meta.history.add_now("LEVEL1-correct_psf", f"PSF corrected with {model_path} model")
    else:
        data_object.meta.history.add_now("LEVEL1-correct_psf", "Empty model path so no correction applied")
        logger.info("No model path so PSF correction is skipped")

    logger.info("correct_psf finished")
    return data_object
