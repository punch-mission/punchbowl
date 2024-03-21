import os

import numpy as np
from prefect import task

from punchbowl.data import PUNCHData


def validate_image_is_square(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Image must be of type np.ndarray. Found: {type(image)}.")
    if len(image.shape) != 2:
        raise ValueError(f"Image must be a 2-D array. Input has {len(image.shape)} dimensions.")
    if not np.equal(*image.shape):
        raise ValueError(f"Image must be square, i.e. same size in both dimensions. Found: {image.shape}.")


@task
def output_image_task(data: PUNCHData, output_filename: str) -> None:
    """Prefect task to write an image to disk

    Parameters
    ----------
    data : PUNCHData
        data that is to be written
    output_filename : str
        where to write the file out

    Returns
    -------
    None
    """
    output_dir = os.path.dirname(output_filename)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    data.write(output_filename)


@task
def load_image_task(input_filename: str) -> PUNCHData:
    """Prefect task to load data for processing

    Parameters
    ----------
    input_filename : str
        path to file to load

    Returns
    -------
    PUNCHData
        loaded version of the image
    """
    return PUNCHData.from_fits(input_filename)
