import typing as t

import numpy as np
from prefect import get_run_logger, task

from punchbowl.data import PUNCHData


def create_coefficient_image(
    flat_coefficients: np.ndarray, image_shape: tuple
) -> np.ndarray:
    """Given a set of coefficients that should apply for every pixel,
        converts them to the required coefficient_image format.

    Parameters
    ----------
    flat_coefficients : np.ndarray
        A one-dimensional list of coefficients that should apply to every pixel in the image.
        Coefficients should be ordered from the highest power to lowest as expected in `photometric_calibration`, e.g.
        f(i,j) = a+b*IMG[i,j]+c*IMG[i,j]^2 would have flat_coefficients of [c, b, a]
    image_shape : tuple
        A tuple of the shape of the image that will be calibrated using `photometric_calibration`

    Returns
    -------
    np.ndarray
        An image of coefficients that apply to every pixel as expected by `photometric_calibration`
    """
    return np.stack(
        [np.ones(image_shape) * coeff for coeff in flat_coefficients], axis=2
    )


def create_constant_quartic_coefficients(img_shape: tuple) -> np.ndarray:
    """Creates a constant coefficients image that preserves the original values,
    i.e. b = 1 and all other coefficients are 0

    Parameters
    ----------
    img_shape : tuple[Int]
        size of the image to create the coefficients for

    Returns
    -------
    np.ndarray
        An image of coefficients that apply to every pixel as expected by `photometric_calibration`
    """
    return create_coefficient_image(np.array([0, 0, 0, 1, 0]), img_shape)


def photometric_calibration(
        image: np.ndarray,
        coefficient_image: np.ndarray
) -> np.ndarray:
    """Computes a non-linear photometric calibration of PUNCH images

    Parameters
    ----------
    image : np.ndarray
        Image to be corrected.

    coefficient_image : np.ndarray
        Frame containing uncertainty values.
        The first two dimensions are the spatial dimensions of the image.
        The last dimension iterates over the powers of the coefficients. Starting with index 0 being the highest power
        and counting down.

    Returns
    -------
    np.ndarray
        a photometrically corrected frame

    Notes
    ------
    Each instrument is subject to an independent non-linear photometric response,
    which needs to be corrected. The module converts from raw camera digitizer
    number (DN) to photometric units at each pixel. Each pixel is replaced with
    the corresponding value of the quartic polynomial in the current CF data
    product for that particular camera.

    A quartic polynomial is applied as follows:

    .. math:: X_{i,j} = a_{i,j}+b_{i,j}*DN_{i,j}+c_{i,j}*DN_{i,j}^2+d_{i,j}*DN_{i,j}^3+e_{i,j}*DN_{i,j}^4

    for each pixel in the detector. Where each quantity (a, b, c, d, e) is a function
    of pixel location (i,j), and is generated using dark current and Stim lamp
    maps. a = offset (dark and the bias). b, b, c, d, e = higher order terms.
    Specifically ``coefficient_image[i,j,:] = [e, d, c, b, a]`` (highest order terms first)

    As each pixel is independent, a quartic fit calibration file (CF) of
    dimensions 2k*2k*5 is constructed, with each layer containing one of the five
    polynomial coefficients for each pixel.

    Examples
    --------
    >>> punch_image = np.ones((100,100))
    >>> coefficient_image = create_coefficient_image(np.array([0, 0, 0, 1, 0]), punch_image.shape)
    >>> data = photometric_calibration(punch_image, coefficient_image)
    """

    # inspect dimensions
    if len(image.shape) != 2:
        raise ValueError("`image` must be a 2-D image")

    if len(coefficient_image.shape) != 3:
        raise ValueError("`coefficient_image` must be a 3-D image")

    if coefficient_image.shape[:-1] != image.shape:
        raise ValueError("`coefficient_image` and `image` must have the same shape`")

    # find the number of quartic fit coefficients
    num_coefficients = coefficient_image.shape[2]
    return np.sum(
        [
            coefficient_image[..., i] * np.power(image, num_coefficients - i - 1)
            for i in range(num_coefficients)
        ],
        axis=0,
    )


@task
def perform_quartic_fit_task(data_object: PUNCHData, quartic_coefficients_path: t.Optional[str] = None) -> PUNCHData:
    """Prefect task to perform the quartic fit calibration on the data

    Parameters
    ----------
    data_object : PUNCHData
        a data object that needs calibration
    quartic_coefficients_path: Optional[str]
        path to a  cube of coefficients as produced by `create_coefficients_image` or `create_ones_coefficients_image`,
        skips correction if it is None

    Returns
    -------
    PUNCHData
        modified version of the input with the quartic fit correction applied

    See Also
    --------
    photometric_calibration

    """
    logger = get_run_logger()
    logger.info("perform_quartic_fit started")

    if quartic_coefficients_path is not None:
        quartic_coefficients = PUNCHData.from_fits(quartic_coefficients_path)
        new_data = photometric_calibration(data_object.data, quartic_coefficients.data)
        data_object = data_object.duplicate_with_updates(data=new_data)
        data_object.meta.history.add_now("LEVEL1-quartic_fit",
                                         f"Quartic fit correction completed with {quartic_coefficients_path}")
    else:
        data_object.meta.history.add_now("LEVEL1-quartic_fit", "Quartic fit correction skipped since path is empty")

    logger.info("perform_quartic_fit finished")

    return data_object
