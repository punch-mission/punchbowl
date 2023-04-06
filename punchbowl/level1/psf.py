import pathlib

from prefect import get_run_logger, task
from regularizepsf.corrector import ArrayCorrector

from punchbowl.data import PUNCHData

THIS_DIRECTORY = pathlib.Path(__file__).parent.resolve()


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
    model_path: pathlib.Path = THIS_DIRECTORY / "data" / "punch_array_corrector.h5",
) -> PUNCHData:
    """ Prefect Task to correct the PSF of an iamge

    Parameters
    ----------
    data_object : PUNCHData
        data to operate on
    model_path : pathlib.Path
        path to the PSF model to use in the correction

    Returns
    -------
    PUNCHData
        modified version of the input with the PSF corrected
    """
    logger = get_run_logger()
    logger.info("correct_psf started")

    # TODO: make pass in object instead of loading from a file
    corrector = ArrayCorrector.load(model_path)
    data_object = correct_psf(data_object, corrector)

    logger.info("correct_psf finished")
    data_object.meta.history.add_now("LEVEL1-correct_psf", "PSF corrected")
    return data_object
