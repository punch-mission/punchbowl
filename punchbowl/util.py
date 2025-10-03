import os
import abc
import warnings
from typing import Generic, TypeVar
from datetime import UTC, datetime, timedelta

import numba
import numpy as np
from dateutil.parser import parse as parse_datetime
from ndcube import NDCube
from astropy.io import fits

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

def bundle_matched_mzp(m_paths, z_paths, p_paths, threshold=75):

    m_dateobs = [parse_datetime(fits.getheader(path, ext=2)['DATE-OBS']) for path in m_paths]
    z_dateobs = [parse_datetime(fits.getheader(path, ext=2)['DATE-OBS']) for path in z_paths]
    p_dateobs = [parse_datetime(fits.getheader(path, ext=2)['DATE-OBS']) for path in p_paths]

    # use Z as the reference
    triplets = []
    for z_index, z_datetime in enumerate(z_dateobs):
        m_deltas = [abs((z_datetime - m_datetime).total_seconds()) for m_datetime in m_dateobs]
        p_deltas = [abs((z_datetime - p_datetime).total_seconds()) for p_datetime in p_dateobs]
        matching_m = np.argmin(m_deltas)
        matching_p = np.argmin(p_deltas)
        if m_deltas[matching_m] > threshold:
            warnings.warn("No matching M")
        elif p_deltas[matching_p] > threshold:
            warnings.warn("No matching P")
        else:
            triplets.append((m_paths[matching_m], z_paths[z_index], p_paths[matching_p]))

    return triplets


T = TypeVar("T")


class DataLoader(abc.ABC, Generic[T]):
    """Interface for passing callable objects instead of file paths to be loaded."""

    @abc.abstractmethod
    def load(self) -> T:
        """Load the data."""

    @abc.abstractmethod
    def src_repr(self) -> str:
        """Return a string representation of the data source."""
