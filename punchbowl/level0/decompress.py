from punchbowl.data import PUNCHData

import numpy as np
from ndcube import NDCube
from prefect import task, get_run_logger
from astropy.wcs import WCS
import astropy.units as u
from typing import Dict, Any

import numpy as np

# TODO : restore default ccd values to functions

# TODO : restore table write to file

# TODO : General cleanup before push

def decode(data, from_bits, to_bits, ccd_gain, ccd_bias, ccd_read_noise):
    """
    Square root decode between specified bitrate values

    Parameters
    ----------
    data
        input encoded data array
    from_bits
        Specified bitrate of encoded image to unpack
    to_bits
        Specified bitrate of output data (decoded)
    ccd_gain
        ccd gain [photons / DN] #TODO - check this
    ccd_offset
        ccd bias level [DN]
    ccd_read_noise
        ccd read noise level [DN]

    Returns
    -------
    np.ndarray
        square root decoded version of the input image

    """

    table = generate_decode_table(from_bits, to_bits, ccd_gain, ccd_bias, ccd_read_noise)

    return decode_by_table(data, table)


def encode(data, from_bits, to_bits):
    """
    Square root encode between specified bitrate values

    Parameters
    ----------
    data
        Input data array
    from_bits
        Specified bitrate of original input image
    to_bits
        Specified bitrate of output encoded image

    Returns
    -------
    np.ndarray
        Encoded version of input data

    """

    data = np.round(data).astype(np.int32).clip(0, None)
    factor = np.array(2 ** (2 * to_bits - from_bits))
    data_scaled_by_factor = np.round(data * factor).astype(np.int32)

    return np.floor(np.sqrt(data_scaled_by_factor)).astype(np.int32)


def decode_simple(data, from_bits, to_bits):
    """
    Performs a simple decoding using the naive squaring strategy

    Parameters
    ----------
    data
        Input data array
    from_bits
        Specified bitrate of original input image
    to_bits
        Specified bitrate of output encoded image

    Returns
    -------
    np.ndarray
        Decoded version of input data

    """

    data = np.round(data).astype(np.int32).clip(0, None)
    factor = 2.0 ** (2 * to_bits - from_bits)

    return np.round(np.square(data) / factor).astype(np.int32)


def noise_pdf(data_value, ccd_gain, ccd_offset, ccd_read_noise, n_sigma=5, n_steps=10000):
    """
    Generates a probability distribution function (pdf) from input an data value

    Parameters
    ----------
    data_value
        Input data value
    ccd_gain
        ccd gain [photons / DN] #TODO - check this
    ccd_offset
        ccd bias level [DN]
    ccd_read_noise
        ccd read noise level [DN]
    n_sigma
        Number of sigma steps
    n_steps
        Number of data steps


    Returns
    -------
    np.ndarray
        Data step distribution
    normal
        Normal distribution

    """

    # Use camera calibration to get an e-count
    electrons = np.clip((data_value - ccd_offset) / ccd_gain, 1, None)

    # Shot noise, converted back to DN
    poisson_sigma = np.sqrt(electrons) * ccd_gain

    # Total sigma is quadrature sum of fixed & shot
    sigma = np.sqrt(poisson_sigma ** 2 + ccd_read_noise ** 2)

    dn_steps = np.arange(-n_sigma * sigma, n_sigma * sigma, sigma * n_sigma * 2 / n_steps)

    # Explicitly calculate the Gaussian/normal PDF at each step
    normal = np.exp(-dn_steps * dn_steps / sigma / sigma / 2)

    # Easier to normalize numerically than to account for missing tails
    normal = normal / np.sum(normal)

    return data_value + dn_steps, normal


def mean_b_offset(data_value, from_bits, to_bits, ccd_gain, ccd_offset, ccd_read_noise):
    """
    Compute an offset from the naive and robust decoding processes

    Parameters
    ----------
    data_value
        input data value [DN]
    from_bits
        Specified bitrate of encoded image to unpack
    to_bits
        Specified bitrate of output data (decoded)
    ccd_gain
        ccd gain [photons / DN] #TODO - check this
    ccd_offset
        ccd bias level [DN]
    ccd_read_noise
        ccd read noise level [DN]

    Returns
    -------
    np.int
        Generated decoding value for use in constructing a decoding table

    """
    naive_decoded_value = decode_simple(data_value, from_bits, to_bits)

    # Generate distribution around naive value
    (values, weights) = noise_pdf(naive_decoded_value, ccd_gain, ccd_offset, ccd_read_noise)

    # Ignore values below the offset -- which break the noise model
    weights = weights * (values >= ccd_offset)

    if np.sum(weights) < 0.95:
        return 0

    weights = weights / np.sum(weights)

    # Encode the entire value distribution
    data_values = encode(values, from_bits, to_bits)

    # Decode the entire value distribution to find the net offset
    net_offset = decode_simple(data_values, from_bits, to_bits)

    # Expected value of the entire distribution
    expected_value = np.sum(net_offset * weights)

    # Return ΔB.
    return expected_value - naive_decoded_value


def decode_corrected(data_value, from_bits, to_bits, ccd_gain, ccd_offset, ccd_read_noise):
    """
    Compute an individual decoding value for an input data value

    Parameters
    ----------
    data_value
        input data value [DN]
    from_bits
        Specified bitrate of encoded image to unpack
    to_bits
        Specified bitrate of output data (decoded)
    ccd_gain
        ccd gain [photons / DN] #TODO - check this
    ccd_offset
        ccd bias level [DN]
    ccd_read_noise
        ccd read noise level [DN]

    Returns
    -------
    np.int
        Generated decoding value for use in constructing a decoding table

    """

    s1p = decode_simple(data_value + 1, from_bits, to_bits)
    s1n = decode_simple(data_value - 1, from_bits, to_bits)

    width = (s1p - s1n) / 4

    fixed_sigma = np.sqrt(ccd_read_noise ** 2 + width ** 2)

    of = mean_b_offset(data_value, from_bits, to_bits, ccd_gain, ccd_offset, fixed_sigma)

    return decode_simple(data_value, from_bits, to_bits) - of


def generate_decode_table(from_bits, to_bits, ccd_gain, ccd_offset, ccd_read_noise):
    """
    Generates a square root decode table between specified bitrate values and CCD parameters

    Parameters
    ----------
    from_bits
        Specified bitrate of encoded image to unpack
    to_bits
        Specified bitrate of output data (decoded)
    ccd_gain
        ccd gain [photons / DN] #TODO - check this
    ccd_offset
        ccd bias level [DN]
    ccd_read_noise
        ccd read noise level [DN]

    Returns
    -------
    table
        Generated square root decoding table

    """

    table = np.zeros(2 ** to_bits)

    for i in range(0, 2 ** to_bits):
        table[i] = decode_corrected(i, from_bits, to_bits, ccd_gain, ccd_offset, ccd_read_noise)

    return table


def decode_by_table(data, table):
    """
    Generates a square root decode table between specified bitrate values and CCD parameters

    Parameters
    ----------
    data
        input encoded data array

    Returns
    -------
    np.ndarray
        Decoded version of input data

    """

    data = np.round(data).astype(np.int32).clip(0, table.shape[0])

    return table[data]


@task
def decompress_data(level0: PUNCHData, path: str) -> Dict[str, Any]:

    logger = get_run_logger()
    logger.info("decompress started")

    # TODO: do decompressing in here

    logger.info("decompress finished")
    data_object.add_history(datetime.now(), "LEVEL0-decompress", "image decompressed")
    return data_object
