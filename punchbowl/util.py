import os
import abc
import warnings
from typing import Generic, TypeVar
from datetime import UTC, datetime

import numba
import numpy as np
from numpy.typing import ArrayLike
from dateutil.parser import parse as parse_datetime
from ndcube import NDCube
from scipy.signal import convolve2d

from punchbowl.data import load_ndcube_from_fits, write_ndcube_to_fits
from punchbowl.exceptions import InvalidDataError, MissingTimezoneWarning
from punchbowl.prefect import punch_task


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


def load_mask_file(path: str) -> np.ndarray:
    """Load a PUNCH instrument mask."""
    with open(path, "rb") as f:
        b = f.read()
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8)).reshape(2048, 2048).T.astype(bool)


@punch_task
def output_image_task(data: NDCube, output_filename: str) -> None:
    """
    Prefect task to write an image to disk.

    Parameters
    ----------
    data : NDCube
        data that is to be written
    output_filename : str
        where to write the file out

    Returns
    -------
    None

    """
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    write_ndcube_to_fits(data, output_filename)


@punch_task(tags=["image_loader"])
def load_image_task(input_filename: str, include_provenance: bool = True, include_uncertainty: bool = True) -> NDCube:
    """
    Prefect task to load data for processing.

    Parameters
    ----------
    input_filename : str
        path to file to load
    include_provenance : bool
        whether to load the provenance layer
    include_uncertainty : bool
        whether to load the uncertainty layer

    Returns
    -------
    NDCube
        loaded version of the image

    """
    return load_ndcube_from_fits(
        input_filename, include_provenance=include_provenance, include_uncertainty=include_uncertainty)


def average_datetime(datetimes: list[datetime]) -> datetime:
    """Compute average datetime from a list of datetimes."""
    timestamps = [dt.replace(tzinfo=UTC).timestamp() for dt in datetimes]
    average_timestamp = sum(timestamps) / len(timestamps)
    return datetime.fromtimestamp(average_timestamp).astimezone(UTC)


@numba.njit(parallel=True, cache=True)
def nan_percentile(array: np.ndarray, percentile: float | list[float]) -> float | np.ndarray:
    """
    Calculate the nan percentile of a 3D cube. Isn't as fast as possible on a single core, but parallelizes very well.

    It's documented that numba's sort is slower than numpy's, and this runs single-threaded ~half as fast as the old
    implementation using numpy. But this parallelizes extremely well, even up to 128 cores for a 1kx2kx2k cube! Thread
    count can be configured by setting numba.config.NUMBA_NUM_THREADS

    The .copy() for each sequence means that, even though percentiling along the zeroth dimension seems wrong from a CPU
    cache standpoint, transposing the input cube makes very little difference (much less than the time cost of copying
    the cube into a transposed orientation!). Disabling the copy for a well-dimensioned array doesn't make a clear
    difference to execution time.

    The nan handling appears to add only negligible computation time
    """
    percentiles = np.atleast_1d(np.array(percentile))
    percentiles = percentiles / 100

    output = np.empty((len(percentiles), *array.shape[1:]))
    for i in numba.prange(array.shape[1]):
        for j in range(array.shape[2]):
            sequence = array[:, i, j].copy()
            n_valid_obs = len(sequence)
            sequence_max = np.nanmax(sequence)
            for index in range(len(sequence)):
                if np.isnan(sequence[index]):
                    sequence[index] = sequence_max
                    n_valid_obs -= 1
            if n_valid_obs == 0:
                for k in range(len(percentiles)):
                    output[k, i, j] = np.nan
            sequence.sort()

            for k in range(len(percentiles)):
                index = (n_valid_obs - 1) * percentiles[k]
                f = int(np.floor(index))
                c = int(np.ceil(index))
                if f == c:
                    output[k, i, j] = sequence[f]
                else:
                    f_val = sequence[f]
                    c_val = sequence[c]
                    output[k, i, j] = f_val + (c_val - f_val) * (index - f)

    if isinstance(percentile, (int, float)):
        return output[0]
    return output


@numba.njit(parallel=True, cache=True)
def parallel_sort_first_axis(array: np.ndarray, handle_nans: bool = False, inplace: bool = False) -> np.ndarray:
    """
    Sorts a 3D cube along the first axis.

    Parallelizes very well on punch190 and phoenix.

    It's documented that numba's sort is slower than numpy's, but this parallelizes extremely well, even up to 64 cores
    for a 1kx2kx2k cube! Thread count can be configured by setting numba.config.NUMBA_NUM_THREADS

    The .copy() for each sequence means that, even though sorting along the zeroth dimension seems wrong from a CPU
    cache standpoint, transposing the input cube makes very little difference (much less than the time cost of copying
    the cube into a transposed orientation!).

    If handle_nans is True, NaNs are explicitly sorted to the high end of the array. Numba's sort appears to do this
    anyway and still sorts the rest of the array correctly, but the flag ensures this behavior with a speed penalty.

    Sorting in-place offers a ~50% speed boost in a 1kx2kx2k test case.
    """
    output = array if inplace else np.empty_like(array)

    for i in numba.prange(array.shape[1]):
        for j in range(array.shape[2]):
            sequence = array[:, i, j].copy()
            if handle_nans:
                bad_val = np.nanmax(sequence) + 1
                for index in range(len(sequence)):
                    if np.isnan(sequence[index]):
                        sequence[index] = bad_val

            sequence.sort()

            if handle_nans:
                for index in range(len(sequence)):
                    if sequence[index] == bad_val:
                        sequence[index] = np.nan

            output[:, i, j] = sequence
    return output


@numba.njit(parallel=True, cache=True)
def nan_percentile_2d(array: np.ndarray, percentile: float | list[float], # noqa: C901
                      window_size: int, preserve_nans: bool = True) -> float | np.ndarray:
    """
    Percentile-filter a 2D cube with NaN awareness. Parallelizes well.

    Each pixel is replaced with a percentile of the non-NaN pixels in a local window. At the image edges, the local
    window is clamped at the image boundary.

    See nan_percentile for performance notes

    When preserve_nans is True, NaN pixels will remain NaN. Otherwise they will be replaced with the percentile value.
    """
    percentiles = np.atleast_1d(np.array(percentile))
    percentiles = percentiles / 100

    half_window_size = window_size // 2

    output = np.empty((len(percentiles), *array.shape))
    for i in numba.prange(array.shape[0]):
        for j in range(array.shape[1]):
            if preserve_nans and np.isnan(array[i, j]):
                for k in range(len(percentiles)):
                    output[k, i, j] = np.nan
                continue
            imin = max(0, i - half_window_size)
            jmin = max(0, j - half_window_size)
            imax = min(array.shape[0], i + half_window_size + 1)
            jmax = min(array.shape[1], j + half_window_size + 1)
            sequence = array[imin:imax, jmin:jmax].flatten()
            n_valid_obs = len(sequence)
            sequence_max = np.nanmax(sequence)
            for index in range(len(sequence)):
                if np.isnan(sequence[index]):
                    sequence[index] = sequence_max
                    n_valid_obs -= 1
            if n_valid_obs == 0:
                for k in range(len(percentiles)):
                    output[k, i, j] = np.nan
                continue
            sequence.sort()

            for k in range(len(percentiles)):
                index = (n_valid_obs - 1) * percentiles[k]
                f = int(np.floor(index))
                c = int(np.ceil(index))
                if f == c:
                    output[k, i, j] = sequence[f]
                else:
                    f_val = sequence[f]
                    c_val = sequence[c]
                    output[k, i, j] = f_val + (c_val - f_val) * (index - f)

    if isinstance(percentile, (int, float)):
        return output[0]
    return output


def interpolate_data(data_before: NDCube, data_after:NDCube, reference_time: datetime, time_key: str = "DATE-OBS",
                     allow_extrapolation: bool = False) -> np.ndarray:
    """Interpolates between two data objects."""
    before_date = parse_datetime(data_before.meta[time_key].value + " UTC").timestamp()
    after_date = parse_datetime(data_after.meta[time_key].value + " UTC").timestamp()
    if reference_time.tzinfo is None:
        warnings.warn("Reference time has no timezone, but should probably be set to UTC", MissingTimezoneWarning)
    observation_date = reference_time.timestamp()

    if before_date > observation_date and not allow_extrapolation:
        msg = "Before data was after the observation date"
        raise InvalidDataError(msg)

    if after_date < observation_date and not allow_extrapolation:
        msg = "After data was before the observation date"
        raise InvalidDataError(msg)

    if before_date == observation_date:
        data_interpolated = data_before
    elif after_date == observation_date:
        data_interpolated = data_after
    else:
        data_interpolated = ((data_after.data - data_before.data)
                              * (observation_date - before_date) / (after_date - before_date)
                              + data_before.data)

    return data_interpolated

def load_spacecraft_mask(path_mask: str) -> np.ndarray:
    """Load the specified spacecraft mask."""
    with open(path_mask, "rb") as f:
        byte_array = f.read()
    mask = np.unpackbits(np.frombuffer(byte_array, dtype=np.uint8)).reshape(2048, 2048)
    return mask.T

def find_first_existing_file(inputs: list[NDCube]) -> NDCube | None:
    """Find the first cube that's not None in a list of NDCubes."""
    for cube in inputs:
        if cube is not None:
            return cube
    msg = "No cube found. All inputs are None."
    raise RuntimeError(msg)

def bundle_matched_mzp(m_cubes: list[NDCube],
                       z_cubes: list[NDCube],
                       p_cubes: list[NDCube],
                       threshold: float = 75.0) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Search and bundle MZP triplets closest in time."""
    m_dateobs = [cube.meta.datetime for cube in m_cubes]
    z_dateobs = [cube.meta.datetime for cube in z_cubes]
    p_dateobs = [cube.meta.datetime for cube in p_cubes]

    # use Z as the reference
    triplets = []
    for z_index, z_datetime in enumerate(z_dateobs):
        m_deltas = [abs((z_datetime - m_datetime).total_seconds()) for m_datetime in m_dateobs]
        p_deltas = [abs((z_datetime - p_datetime).total_seconds()) for p_datetime in p_dateobs]
        matching_m = np.argmin(m_deltas)
        matching_p = np.argmin(p_deltas)
        m_time_diff = m_deltas[matching_m]
        p_time_diff = p_deltas[matching_p]

        if m_time_diff > threshold or p_time_diff > threshold:
            missing = []
            if m_time_diff > threshold:
                missing.append("M")
            if p_time_diff > threshold:
                missing.append("P")
            msg = f"No matching {' and '.join(missing)} for Z at {z_datetime.isoformat()}"
            warnings.warn(msg)
        else:
            triplets.append((m_cubes[matching_m], z_cubes[z_index], p_cubes[matching_p]))
    return triplets

def masked_mean(data: ArrayLike,
                mask: ArrayLike)-> np.ndarray:
    """Masked nanmean with entries where both mask is True and data is finite."""
    valid = mask & np.isfinite(data)
    count = valid.sum(axis=0)

    sumvalid = np.where(valid, data, 0.0).sum(axis=0)

    # Safe divide; pixels with count==0 become NaN
    outdata = np.full(sumvalid.shape, np.nan, dtype=np.result_type(data, np.float32))
    np.divide(sumvalid, count, out=outdata, where=(count > 0))
    return outdata

T = TypeVar("T")


class DataLoader(abc.ABC, Generic[T]):
    """Interface for passing callable objects instead of file paths to be loaded."""

    @abc.abstractmethod
    def load(self) -> T:
        """Load the data."""

    @abc.abstractmethod
    def src_repr(self) -> str:
        """Return a string representation of the data source."""

def inpaint_nans(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Fill nans in an image with a neighborhood value.

    Parameters
    ----------
    image : np.ndarray
        image with nans
    kernel_size : int
        odd integer size for the smoothing kernel

    Returns
    -------
    np.ndarray
        image with nans filled

    """
    image = image.copy()  # don't mutate the original image

    if kernel_size % 2 == 0:
        msg = "Kernel size must be odd."
        raise RuntimeError(msg)
    kernel = np.ones((kernel_size, kernel_size))
    kernel[kernel_size//2, kernel_size//2] = 0
    last_nan_mask = np.zeros(image.shape, dtype=bool)
    while np.any(np.isnan(image)):
        nan_mask = np.isnan(image)
        if np.all(nan_mask == last_nan_mask):
            # Nothing's changed, so let's bail out. This can happen if an image has corrupted packets, causing every
            # row to pass the row threshold and thus every pixel is NaN
            break
        last_nan_mask = nan_mask
        image[nan_mask] = 0
        neighbors = convolve2d(~nan_mask, kernel, mode="same", boundary="symm")
        convolved = convolve2d(image, kernel, mode="same", boundary="symm")
        convolved[neighbors>0] = convolved[neighbors>0]/neighbors[neighbors>0]
        convolved[neighbors==0] = np.nan
        convolved[~nan_mask] = image[~nan_mask]
        image = convolved
    return image
