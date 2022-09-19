import numpy as np


def validate_image_is_square(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Image must be of type np.ndarray. Found: {type(image)}.")
    if len(image.shape) != 2:
        raise ValueError(f"Image must be a 2-D array. Input has {len(image.shape)} dimensions.")
    if not np.equal(*image.shape):
        raise ValueError(f"Image must be square, i.e. same size in both dimensions. Found: {image.shape}.")
