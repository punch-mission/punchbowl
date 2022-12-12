from typing import Dict, Tuple
from datetime import datetime
import pathlib

from prefect import task, get_run_logger
import numpy as np
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

    # TODO: needs to copy not just the data but all the meta
    return PUNCHData(data=new_data)


@task
def correct_psf_task(
    data_object: PUNCHData,
    model_path: pathlib.Path = THIS_DIRECTORY / "data" / "punch_array_corrector.h5",
) -> PUNCHData:
    logger = get_run_logger()
    logger.info("correct_psf started")

    corrector = ArrayCorrector.load(model_path)
    data_object = correct_psf(data_object, corrector)

    logger.info("correct_psf finished")
    data_object.add_history(datetime.now(), "LEVEL1-correct_psf", "PSF corrected")
    return data_object
