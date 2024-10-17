import os.path

import numpy as np
from ndcube import NDCube
from prefect import get_run_logger, task

TABLE_PATH = os.path.dirname(__file__) + "/decoding_tables/"


def decode_sqrt(
    data: np.ndarray | float,
    from_bits: int = 16,
    to_bits: int = 12,
    ccd_gain: float = 1 / 4.3,
    ccd_offset: float = 100,
    ccd_read_noise: float = 17,
    overwrite_table: bool = False,
) -> np.ndarray:
    """
    Square root decode between specified bitrate values.

    Parameters
    ----------
    data
        Input encoded data array
    from_bits
        Specified bitrate of encoded image to unpack
    to_bits
        Specified bitrate of output data (decoded)
    ccd_gain
        CCD gain [photons / DN]
    ccd_offset
        CCD bias level [DN]
    ccd_read_noise
        CCD read noise level [DN]
    overwrite_table
        Toggle to regenerate and overwrite existing decoding table

    Returns
    -------
    np.ndarray
        Square root decoded version of the input image

    """
    table_name = (
        TABLE_PATH
        + "tab_fb"
        + str(from_bits)
        + "_tb"
        + str(to_bits)
        + "_g"
        + str(1 / ccd_gain)
        + "_b"
        + str(ccd_offset)
        + "_r"
        + str(ccd_read_noise)
        + ".npy"
    )

    if not os.path.isfile(table_name) or overwrite_table:
        table = generate_decode_sqrt_table(from_bits, to_bits, ccd_gain, ccd_offset, ccd_read_noise)

        if not os.path.isdir(TABLE_PATH):
            os.makedirs(TABLE_PATH, exist_ok=True)

        np.save(table_name, table)
    else:
        table = np.load(table_name)

    return decode_sqrt_by_table(data, table)


def encode_sqrt(data: np.ndarray | float, from_bits: int = 16, to_bits: int = 12) -> np.ndarray:
    """
    Square root encode between specified bitrate values.

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


def decode_sqrt_simple(data: np.ndarray | float, from_bits: int = 16, to_bits: int = 12) -> np.ndarray:
    """
    Perform a simple decoding using the naive squaring strategy.

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


def noise_pdf(
    data_value: np.ndarray | float,
    ccd_gain: float = 1 / 4.3,
    ccd_offset: float = 100,
    ccd_read_noise: float = 17,
    n_sigma: int = 5,
    n_steps: int = 10000,
) -> tuple:
    """
    Generate a probability distribution function (pdf) from an input data value.

    Parameters
    ----------
    data_value
        Input data value
    ccd_gain
        CCD gain [DN / electron]
    ccd_offset
        CCD bias level [DN]
    ccd_read_noise
        CCD read noise level [DN]
    n_sigma
        Number of sigma steps
    n_steps
        Number of data steps


    Returns
    -------
    np.ndarray
        Data step distribution
    normal
        Data normal distribution

    """
    # Use camera calibration to get an e-count
    electrons = np.clip((data_value - ccd_offset) / ccd_gain, 1, None)

    # Shot noise, converted back to DN
    poisson_sigma = np.sqrt(electrons) * ccd_gain

    # Total sigma is quadrature sum of fixed & shot
    sigma = np.sqrt(poisson_sigma**2 + ccd_read_noise**2)

    dn_steps = np.arange(-n_sigma * sigma, n_sigma * sigma, sigma * n_sigma * 2 / n_steps)

    # Explicitly calculate the Gaussian/normal PDF at each step
    normal = np.exp(-dn_steps * dn_steps / sigma / sigma / 2)

    # Easier to normalize numerically than to account for missing tails
    normal = normal / np.sum(normal)

    return data_value + dn_steps, normal


def mean_b_offset(
    data_value: float,
    from_bits: int = 16,
    to_bits: int = 12,
    ccd_gain: float = 1 / 4.3,
    ccd_offset: float = 100,
    ccd_read_noise: float = 17,
) -> float:
    """
    Compute an offset from the naive and robust decoding processes.

    Parameters
    ----------
    data_value
        Input data value [DN]
    from_bits
        Specified bitrate of encoded image to unpack
    to_bits
        Specified bitrate of output data (decoded)
    ccd_gain
        CCD gain [DN / electron]
    ccd_offset
        CCD bias level [DN]
    ccd_read_noise
        CCD read noise level [DN]

    Returns
    -------
    float
        Generated decoding value for use in constructing a decoding table

    """
    naive_decoded_value = decode_sqrt_simple(data_value, from_bits, to_bits)

    # Generate distribution around naive value
    (values, weights) = noise_pdf(naive_decoded_value, ccd_gain, ccd_offset, ccd_read_noise)

    # Ignore values below the offset -- which break the noise model
    weights = weights * (values >= ccd_offset)

    if np.sum(weights) < 0.95:
        return 0

    weights = weights / np.sum(weights)

    # Encode the entire value distribution
    data_values = encode_sqrt(values, from_bits, to_bits)

    # Decode the entire value distribution to find the net offset
    net_offset = decode_sqrt_simple(data_values, from_bits, to_bits)

    # Expected value of the entire distribution
    expected_value = np.sum(net_offset * weights)

    # Return ΔB.
    return expected_value - naive_decoded_value


def decode_sqrt_corrected(
    data_value: float,
    from_bits: int = 16,
    to_bits: int = 12,
    ccd_gain: float = 1 / 4.3,
    ccd_offset: float = 100,
    ccd_read_noise: float = 17,
) -> float:
    """
    Compute an individual decoding value for an input data value.

    Parameters
    ----------
    data_value
        Input data value [DN]
    from_bits
        Specified bitrate of encoded image to unpack
    to_bits
        Specified bitrate of output data (decoded)
    ccd_gain
        CCD gain [DN / electron]
    ccd_offset
        CCD bias level [DN]
    ccd_read_noise
        CCD read noise level [DN]

    Returns
    -------
    float
        Generated decoding value for use in constructing a decoding table

    """
    s1p = decode_sqrt_simple(data_value + 1, from_bits, to_bits)
    s1n = decode_sqrt_simple(data_value - 1, from_bits, to_bits)

    width = (s1p - s1n) / 4

    fixed_sigma = np.sqrt(ccd_read_noise**2 + width**2)

    of = mean_b_offset(data_value, from_bits, to_bits, ccd_gain, ccd_offset, fixed_sigma)

    return decode_sqrt_simple(data_value, from_bits, to_bits) - of


def generate_decode_sqrt_table(
    from_bits: int = 16,
    to_bits: int = 12,
    ccd_gain: float = 1 / 4.3,
    ccd_offset: float = 100,
    ccd_read_noise: float = 17,
) -> np.ndarray:
    """
    Generate a square root decode table between specified bitrate values and CCD parameters.

    Parameters
    ----------
    from_bits
        Specified bitrate of encoded image to unpack
    to_bits
        Specified bitrate of output data (decoded)
    ccd_gain
        CCD gain [DN / electron]
    ccd_offset
        CCD bias level [DN]
    ccd_read_noise
        CCD read noise level [DN]

    Returns
    -------
    table
        Generated square root decoding table

    """
    table = np.zeros(2**to_bits)

    for i in range(2**to_bits):
        table[i] = decode_sqrt_corrected(i, from_bits, to_bits, ccd_gain, ccd_offset, ccd_read_noise)

    return table


def decode_sqrt_by_table(data: np.ndarray | float, table: np.ndarray) -> np.ndarray:
    """
    Generate a square root decode table between specified bitrate values and CCD parameters.

    Parameters
    ----------
    data
        Input encoded data array
    table
        Square root decoding table

    Returns
    -------
    np.ndarray
        Decoded version of input data

    """
    data = np.round(data).astype(np.int32).clip(0, table.shape[0])

    return table[data]


@task
def decode_sqrt_data(data_object: NDCube, overwrite_table: bool = False) -> NDCube:
    """
    Prefect task in the pipeline to decode square root encoded data.

    Parameters
    ----------
    data_object : NDCube
        the object you wish to decode
    overwrite_table
        Toggle to regenerate and overwrite existing decoding table

    Returns
    -------
    NDCube
        a modified version of the input with the data square root decoded

    """
    logger = get_run_logger()
    logger.info("square root decoding started")

    data = data_object.data

    from_bits = data_object.meta["RAWBITS"].value
    to_bits = data_object.meta["COMPBITS"].value

    ccd_gain = data_object.meta["GAINCMD"].value
    ccd_offset = data_object.meta["OFFSET"].value
    ccd_read_noise = 17  # DN  # TODO: make this not a hardcoded value!

    decoded_data = decode_sqrt(
        data,
        from_bits=from_bits,
        to_bits=to_bits,
        ccd_gain=ccd_gain,
        ccd_offset=ccd_offset,
        ccd_read_noise=ccd_read_noise,
        overwrite_table=overwrite_table,
    )
    data_object.data[...] = decoded_data[...]  # TODO: we need to make sure the data type changes?

    logger.info("square root decoding finished")
    data_object.meta.history.add_now("LEVEL0-decode-sqrt", "image square root decoded")

    return data_object
