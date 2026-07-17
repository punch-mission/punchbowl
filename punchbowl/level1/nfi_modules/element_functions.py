import numpy as np
from scipy.special import voigt_profile

from punchbowl.level1.nfi_modules.util import multivector_matrix_multiply

tiny = 1.0e-4


def bin_function(ptcoords, coordarr, params):
    """
    This is a basis function that's a set of rectilinear 'in' or 'out' boxes -- e.g., pixels in 2D:
    
    Parameters
    ----------
    ptcoords:

    coordarr: 

    params
    """
    
    
    return np.prod(np.round((ptcoords - coordarr).T + tiny) == 0, axis=0)

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
    """
    dxa = coords - pt
    q = inputs[0]
    return np.exp(-0.5 * np.sum(dxa * multivector_matrix_multiply(q[0 : pt.size, 0 : pt.size], dxa), axis=-1))

def nd_powgaussian_psf(pt, coords, inputs):
    dxa = coords - pt
    q = inputs[0]
    exp = inputs[1]
    return np.exp(-0.5 * np.sum(dxa * multivector_matrix_multiply(q[0 : pt.size, 0 : pt.size], dxa), axis=-1) ** exp)