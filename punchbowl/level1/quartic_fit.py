import numpy as np
from typing import Optional
from datetime import datetime
from punchbowl.data import PUNCHData
from prefect import task, get_run_logger


def photometric_calibration(image: np.ndarray,
                                quartic_coefficient_map: np.ndarray,
                                uncertainty_map: Optional[np.ndarray]=None) -> np.ndarray:
    """Computes a non-linear photometric callibration of PUNCH images

    Each instrument is subject to an independent non-linear photometric response,
    which needs to be corrected. The module converts from raw camera digitizer 
    number (DN) to photometric units at each pixel. Each pixel is replaced with 
    the corresponding value of the quartic polynomial in the current CF data 
    product for that particular camera. 

    A quartic polynomial (flat field) of the form:
        DN[i,j]=a+b*X[i,j]+c*X[i,j]^2+d*X[i,j]^3+e*X[i,j]^4

    is derived for each pixel in the detector. Where each quantity is a function 
    of pixel location [i,j], and is generated using dark current and Stim lamp 
    maps. Where:
        - a = offset (dark and the bias).
        - b = brightness in pixel.

    As each pixel is independent, a quartic fit calibration file (CF) of 
    dimensions 2k*2k*5 is constructed, with each layer containing one of the five 
    polynomial coefficients for each pixel.

    Parameters
    ----------
    data_frame
        image to be corrected
    uncertainty_frame
        frame containing uncertainty values
    correction_map
        a CF correction map

    Returns
    -------
    np.ndarray
        a photmetrically corected frame

    # TODO : add example call
    """

    # inspect dimensions
    assert len(image.shape) == 2, "function:photometric_calibration, data frame must be a 2-D image"

    # inspect dimensions of correction map and data_frame
    assert quartic_coefficient_map.shape[0] == image.shape[0], "function:photometric_calibration, CF calibration x dim != data frame x dim"
    assert quartic_coefficient_map.shape[1] == image.shape[1], "function:photometric_calibration, CF calibration y dim != data frame y dim"

    # find the number of quartic fit coefficients
    num_coeffs=quartic_coefficient_map.shape[2]

    corrected_data = np.sum([quartic_coefficient_map[:,:,iStep] * np.power(image, num_coeffs-1-iStep) for iStep in range(num_coeffs)], axis=0) 

@task
def perform_quartic_fit(data_object):
    logger = get_run_logger()
    logger.info("perform_quartic_fit started")
    # do quartic fit correction in here
    logger.info("perform_quartic_fit finished")
    data_object.add_history(datetime.now(), "LEVEL1-perform_quartic_fit", "Quartic fit correction completed")
    return data_object

    return corrected_data

