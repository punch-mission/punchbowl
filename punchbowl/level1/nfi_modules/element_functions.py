import numpy as np
from scipy.special import voigt_profile

from punchbowl.level1.nfi_modules.util import multivector_matrix_multiply

TINY = 1.0e-4


def bin_function_evaluator(point_coordinates, coordinate_array, params=None):
    """
    This is a basis function that's a set of rectilinear 'in' or 'out' boxes -- e.g., pixels in 2D:
    
    The difference between the two coordinate arrays is rounded to the
    nearest integer (with a small offset `tiny` added before rounding to
    avoid floating-point ties landing exactly on 0.5) and compared to
    zero, elementwise. Taking the product along axis 0 acts as a logical
    across coordinate dimensions, so a point is flagged as a match
    only if *every* dimension's rounded difference is exactly zero.

    Parameters
    ----------
    point_coordinates: np.ndarray
        Reference coordinates (for a single point) to compare against `coordinate_array`
    coordinate_array: np.ndarray
        Coordinates to test for equality with `point_coordinates`. 

        Likely an array given by `CoordGrid.get_coordinates_from_indices()`.
    params: None
        Not used in this function. 
        Included parameter to meet the function requirements for `ElementGrid.function_evaluator`.
    
    Returns
    -------
    np.ndarray
        Array of 1s and 0s (usable as a boolean mask) where 1 indicates that all coordinate dimensions
        of the corresponding point in `coordinate_array` match `point_coordinates` (within rounding tolerance), 
        and 0 indicates at least one dimension differs.
    """
    return np.prod(np.round((point_coordinates - coordinate_array).T + TINY) == 0, axis=0)

def get_2d_covariance(sigmas, theta):
    """
    Builds covariance matrix for 2D case using Mahalanobis distance.

    Parameters
    ----------
    sigmas : array-like, shape(2,)
        Standard deviations
    theta : float
        Angle (in radians), of the first principal axis relative to the x-axis.

    Returns
    -------
    np.ndarray
        Covariance matrix
    Notes
    -----
    This is a setup function for a 3D or 2D Gaussian type PSF/response function.
    It's defined in terms of the 3 axes of the ellipse and a set of three angles
    about which the ellipse is rotated. The initial axes are x, y, z, while
    the rotation matrices use the Tait Bryan convention z, x, y -- i.e., if
    the angles are all zero, sigmas[0] will be the ellipse length along x,
    sigmas[1] will be the ellipse length along y, etc; and, if the angles are not zero
    the ellipse is first rotated about the z axis (in the x-y plane), then x (z-y plane)
    then y (z-x plane). If the PSF is evaluated in 2D, only the first two
    axis lengths and the first angle will be used.
    """
    vec0 = sigmas[0] * np.array([np.cos(theta), np.sin(theta)])
    vec1 = sigmas[1] * np.array([-np.sin(theta), np.cos(theta)])
    return np.outer(vec0, vec0) + np.outer(vec1, vec1)


def n_dimensional_gaussian_psf(pt, coords, inputs):
    """
    Evaluate an n-dimensional Gaussian PSF centered at the point pt, for each of the
    coordinates coords, based on the q (inverse of covariance) matrix q.
    Coords must have dimensions npts by nd where nd is the dimensionality
    of the Gaussian. q can be larger than nd by nd; higher dimensions will be ignored.

    Parameters
    ----------
    pt: np.ndarray
        Coordinates for point of interest.
    coords: np.ndarray
        Array of coordinates as given by `CoordGrid.get_coordinates_from_indices()` for the 
        relevant coordinate grid of interest.
    inputs: np.ndarray
        An np.ndarray where the first element includes the precision matrix (i.e. the inverse of the 
        covariance).
        Designed to fit the requirements to call as a callable function for `ElementGrid.function_evaluator`
    
    Returns
    -------
    np.ndarray
        Gaussian PSF centered at point `pt`
    """
    displacement_vectors = coords - pt
    precision_matrix = inputs[0]
    return np.exp(-0.5 * np.sum(displacement_vectors * multivector_matrix_multiply(precision_matrix[0 : pt.size, 0 : pt.size], displacement_vectors), axis=-1))