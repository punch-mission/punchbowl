import numpy as np
from nfi_modules.util import multivector_matrix_multiply
from scipy.special import voigt_profile

tiny = 1.0e-4

def bin_function(ptcoords,coordarr,params):
    """This is a basis function that's a set of rectilinear 'in' or 'out' boxes -- e.g., pixels in 2D:"""
    return np.prod(np.round((ptcoords-coordarr).T+tiny) == 0,axis=0)


def spike_function(ptcoords,coordarr,params):
    """
    Rectilinear spike function. Can be 1-D or multidimensional. Should work well for a
    linear (or bilinear for 2D, etc) interpolating basis function. Note: Test this...
    """
    return np.prod(np.clip(1.0-np.abs(ptcoords-coordarr).T,0,1),axis=0)


def get_3d_covariance(sigmas,angles): # Get covariance for 3D Mahalanobis distance:
    """
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
    [[c1,c2,c3],[s1,s2,s3]] = [np.cos(angles),np.sin(angles)]
    # Compute covariance using Tait-Bryan angles ordered z, x, y:
    vec0 = sigmas[0]*np.array([c1*c3-s1*s2*s3, c3*s1+c1*s2*s3, -c2*s3])
    vec1 = sigmas[1]*np.array([-c2*s1, c1*c2, s2])
    vec2 = sigmas[2]*np.array([c1*s3+c3*s1*s2, s1*s3-c1*c3*s2, c2*c3])
    return np.outer(vec0,vec0)+np.outer(vec1,vec1)+np.outer(vec2,vec2)

def get_2d_covariance(sigmas,theta):
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
    vec0 = sigmas[0]*np.array([np.cos(theta),np.sin(theta)])
    vec1 = sigmas[1]*np.array([-np.sin(theta),np.cos(theta)])
    return np.outer(vec0,vec0)+np.outer(vec1,vec1)

def spice_spectrograph_psf(pt, coords, inputs):
    slitwid = inputs[-1]
    if(len(inputs)<4): 
        n_slit_subpts = 5
    else: 
        n_slit_subpts = inputs[-2]
    subpts = np.zeros([n_slit_subpts,2])
    subpts[:,1] = slitwid*np.arange(n_slit_subpts)/n_slit_subpts - 0.5*slitwid + 0.5*slitwid/n_slit_subpts
    psf = nd_powgaussian_psf(pt+subpts[0], coords, inputs)
    for i in range(1,n_slit_subpts): 
        psf += nd_powgaussian_psf(pt+subpts[i], coords, inputs)
    return psf/n_slit_subpts

def n_dimensional_gaussian_psf(pt, coords, inputs):
    """
    Evaluate an n-dimensional Gaussian PSF centered at the point pt, for each of the
    coordinates coords, based on the q (inverse of covariance) matrix q.
    Coords must have dimensions npts by nd where nd is the dimensionality
    of the Gaussian. q can be larger than nd by nd; higher dimensions will be ignored.
    """
    dxa = coords-pt
    q = inputs[0]
    return np.exp(-0.5*np.sum(dxa*multivector_matrix_multiply(q[0:pt.size,0:pt.size],dxa), axis=-1))

def nd_voigt_psf(pt, coords, inputs):
    dxa = coords-pt
    q = inputs[0]
    g = inputs[1]*(np.log(2))**0.5
    e = inputs[2]
    mdist = np.sum(dxa*multivector_matrix_multiply(q[0:pt.size,0:pt.size],dxa), axis=-1)
    return voigt_profile(mdist**0.5,e,g)*(1.0/voigt_profile(0,e,g))


def nd_powgaussian_psf(pt, coords, inputs):
    dxa = coords-pt
    q = inputs[0]
    exp = inputs[1]
    return np.exp(-0.5*np.sum(dxa*multivector_matrix_multiply(q[0:pt.size,0:pt.size],dxa), axis=-1)**exp)

def flattop_guassian_psf(pt, coords, inputs):
    dxa = coords-pt
    q = inputs[0]
    flatness = inputs[1]
    flatsds = inputs[2]
    mdist = np.sum((dxa*multivector_matrix_multiply(q[0:pt.size,0:pt.size],dxa)), axis=-1)
    return (np.exp(-0.5*mdist)/(1.0-0.5*flatness*mdist*(np.exp(-0.5*mdist/(flatsds**2)))**2))
