import copy

import numba
import numpy as np


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
    # TODO: (JK note) I'm pretty sure this is the same as np.matvec() [Issue #1113]
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
