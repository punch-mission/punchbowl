import os

import numpy as np
from ndcube import NDCube
from prefect import task

from punchbowl.data import load_ndcube_from_fits, write_ndcube_to_fits


def validate_image_is_square(image: np.ndarray) -> None:
    """Check that the input array is square."""
    if not isinstance(image, np.ndarray):
        msg = f"Image must be of type np.ndarray. Found: {type(image)}."
        raise TypeError(msg)
    if len(image.shape) != 2:
        msg = f"Image must be a 2-D array. Input has {len(image.shape)} dimensions."
        raise ValueError(msg)
    if not np.equal(*image.shape):
        msg = f"Image must be square. Found: {image.shape}."
        raise ValueError(msg)


@task
def output_image_task(data: NDCube, output_filename: str) -> None:
    """
    Prefect task to write an image to disk.

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
    write_ndcube_to_fits(data, output_filename)


@task
def load_image_task(input_filename: str) -> NDCube:
    """
    Prefect task to load data for processing.

    Parameters
    ----------
    input_filename : str
        path to file to load

    Returns
    -------
    PUNCHData
        loaded version of the image

    """
    return load_ndcube_from_fits(input_filename)
