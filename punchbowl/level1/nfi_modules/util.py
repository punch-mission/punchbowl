import copy

import numba
import numpy as np


@numba.jit(fastmath=True, parallel=True, forceobj=True)
def _masked_medfilt_inner(
    flatinds, data, footprint_inds, footprint_ind_offset, tparg, dat_pad_shape, data_fppad, mask_fppad, data_filt
):
    for ind in flatinds:
        ijkpad = np.unravel_index(ind, data.shape) + footprint_inds - footprint_ind_offset
        ijkpad = np.ravel_multi_index(ijkpad.transpose(tparg), dat_pad_shape)
        dat = data_fppad[ijkpad]
        good = mask_fppad[ijkpad]
        if np.sum(good) > 0:
            data_filt[ind] = np.median(dat[good])
    return data_filt


def multivector_matrix_multiply(a, b):
    """
    Multiply a matrix with each element of a set of vectors.
    (The same as `numpy.matvec`)

    Notes
    -----
    The vectors must be numpy arrays dimensioned nvec by ndim, where ndim is the
    dimensionality of the space, while the matrix is ndim by ndim. This applies in
    general for other operations of a set of vectors with a single vector --
    i.e., the vector space index must be the last one.

    For instance, to add a vector shift to a set of vectors, the array of
    vectors must also be nvec by ndim. Then they can be added as c = a+b. See
    https://numpy.org/doc/stable/user/basics.broadcasting.html

    This is somewhat backward to how dot products otherwise work in numpy.
    So the order of operations has to be reversed compared to normal and the
    matrix must be transposed. i.e., instead of v2 = np.dot(fwd,v1), it's necessary
    to instead do v2 = np.dot(v1,fwd.T).
    There may be a built in numpy way to do this, but I haven't found it so far. `linalg.multi_dot`
    doesn't appear to function any differently from dot in this case. I've implemented it
    here as a very small subroutine rather than spreading it all over the code
    for ease of maintenance and explanation. It works for single vectors, too.
    """
    # TODO: (JK note) I'm pretty sure this is the same as np.matvec()
    return np.dot(b, a.T)


def forward_rolling_transpose(arr):
    """
    Forward rolling transpose.

    for switching from coordinate dimension last to coordinate dimension first in multidimensional coordinate arrays

    Parameters
    ---------
    arr: np.ndarray
        array of interest to forward transpose and roll forward by one.

    Returns
    -------
    np.ndarray
        A new view of `arr` with the axes rotated that the original last axis is now axis 0.

    """
    return arr.transpose(np.roll(np.arange(arr.ndim), 1))


def backward_rolling_transpose(arr):
    """
    Backward rolling transpose.

    for switching from coordinate dimension first to coordinate dimension last in multidimensional
    coordinate arrays

    Essentially the inverse function of `forward_rolling_transpose`.

    Parameters
    ----------
    arr: np.ndarray
        array of interest to backward transpose

    Returns
    -------
    np.ndarray
        A new view of `arr` with the axes rotated so that the original first axis
        is now the last axis.
    """
    return arr.transpose(np.roll(np.arange(arr.ndim), -1))


def roll_transpose_from_numpy_indices(dims, **kwargs):
    """
    Roll transpose.

    Essentially perform `backward_rolling_transpose` on numpy `indices` for given dimensions (`dims`).

    Parameters
    ----------
    dims : sequence of ints
        (same as `dimensions` for `np.indices`) The shape of the grid.

    Returns
    -------
    np.ndarray
        `np.indices` of a grid but the first axis is now the last axis.

    Notes
    -----
    Numpy's indices method is extremely useful, but it puts the coordinate
    dimension (e.g., ijk) first, but for easy vector operations it should be last.
    Transposing puts the coordinate dimension last but it also reverses all of
    the other dimensions, which gets super confusing. This does a `roll' transpose
    which just shifts the dimensions forward by 1. Very simple, but you can see how
    it could get unggkljhly real quick
    """
    ia = np.indices(dims, **kwargs)
    return backward_rolling_transpose(ia)


def bindown(data, out_shape):
    """
    Downsample an N-dimensional array by summing values into coarser bins.

    Parameters
    ----------
    data: np.ndarray
        Input array of arbitrary dimensionality to be rebinned.
    out_shape: tuple of int
        Desired shape of the output array.

    Returns
    -------
    np.ndarray
        Downsampled data with shape `out_shape`.
    """
    inds = np.ravel_multi_index(np.floor(np.indices(data.shape).T * out_shape / np.array(data.shape)).T.astype(np.uint32), out_shape)
    return np.bincount(inds.flatten(), weights=data.flatten(), minlength=np.prod(out_shape)).reshape(out_shape)

def binup(data, factor):
    """
    Upsample an N-dimensional array by repeating values (nearest-neighbor).

    Parameters
    ----------
    data: np.ndarray
        Input array of arbitrary dimensionality to be upsampled.
    factor: float or array-like of float
        Per-axis scale factor(s) describing how much to expand each dimension.
        Rounded to the nearest integer before computing the output shape, so `factor` need not be an exact integer.

    Returns
    -------
    np.ndarray
        Data array upsampled by given `factor`.
    """
    n = np.round(np.array(data.shape) * np.round(factor)).astype(np.int32)
    inds = np.ravel_multi_index(np.floor(np.indices(n).T / np.array(factor)).T.astype(np.uint32), data.shape)
    return np.reshape(data.flatten()[inds], n)
