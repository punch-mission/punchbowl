import os
from sys import path

import numpy as np
from scipy.ndimage import zoom

file_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(file_dir)
module_path = os.path.join(src_dir,'nfi_modules')
path.append(module_path)

from nfi_modules.fwdmats import generate_nfi_fwdmats
from nfi_modules.reconstruct import reconstruct_nfi_straylight
from nfi_modules.util import bindown

from punchbowl.data.punchcube import PUNCHCube


def get_bin_down_crval(crval,cdelt,bin_factor:int):
    """
    Calculates new sky coordinate value of reference pixel based on bin factor.

    Parameters
    ----------
    crval : float
        "crval" or reference pixel (in sky coordinates) of interest 
    cdelt : float
        "cdelt"  (or the size of a pixel in sky coordinates) associated with "crval"
    bin_factor : int
        Binning down factor

    Returns
    -------
    coordinate : float
        The new binned down reference pixel coordinate
    """
    return (-crval/cdelt)/bin_factor

def get_fwd_mat_inputs(datacube: PUNCHCube,
                       bin_factor:int):
    """
    Calculate and return all of the inputs needed for generating the forward matrices for NFI.

    Specifically this returns all the parameters needed as input for the `generate_nfi_fwdmats`  
    function.

    This calculates the appropriately binned down reference pixel coordinates ("crval" in PUNCHCube wcs), and saves the 
    rotation angle ("crota" in PUNCHCube meta data) in radians.

    Parameters
    ----------
    datacube : PUNCHcube
        PUNCH data of interest
    bin_factor : int
        Binning down factor

    Returns
    -------
    x_offsets_binned : float
        The new binned down x-coordinate of the reference pixel ("crval1")
    y_offsets_binned : float
        The new binned down y-coordinate of the reference pixel ("crval2")
    crota_radians : float
        The rotation angle ("CROTA") in units of radians 
    """
    data_crval1, data_crval2 = datacube.wcs.wcs.crval
    data_cdelt1, data_cdelt2 = datacube.wcs.wcs.cdelt

    x_offsets_binned = get_bin_down_crval(data_crval1,data_cdelt1,bin_factor)
    y_offsets_binned = get_bin_down_crval(data_crval2,data_cdelt2,bin_factor)

    crota_radians = datacube.meta['CROTA'].value*np.pi/180  # save the rotation angle in radians

    return x_offsets_binned, y_offsets_binned, crota_radians

def generate_glint_mask(data_shape,
               sphere1_center:tuple = (540, 790),
               sphere2_center:tuple = (540, 1210),
               sphere_radius:int = 375,
               bottom_cut_off:int = 250):
    """
    Create (boolean) mask for circular glints on bottom half of NFI data. 
    (For now the plan is just to mask out the sphere pattern until we have a better way to prefilter it.)

    This creates two circular masks, given estimated sphere centers and radius (assumed to have the same radius). 
    Also masks out a portion of the bottom of the image where the spheres don't reach the bottom of the image.

    Parameters
    ----------
    data_shape : tuple
        Shape of the image data 
    sphere1_center : tuple
        Coordinate for the center of sphere 1 
    sphere2_center : tuple
        Coordinate for the center of sphere 2
    sphere_radius : int
        Radi of sphere (same radi for both spheres)
    bottom_cut_off : int
        the height to which include everything above and mask out everything below

    Returns
    -------
    mask : np.array
        Boolean mask for masking out glint spheres
    """
    x_axis, y_axis = np.indices(data_shape)
    mask =  ((((x_axis-sphere1_center[0])**2+(y_axis-sphere1_center[1])**2)**0.5 > sphere_radius) 
             *((((x_axis-sphere2_center[0])**2+(y_axis-sphere2_center[1])**2)**0.5 > sphere_radius))
             *(x_axis>bottom_cut_off))

    return mask

def get_solver_inputs(datacube: PUNCHCube,
                      glint_mask: np.array,
                      bindown_shape: tuple = (512, 512)):
    """
    Calculate the inputs needed to create the light solutions for NFI, and appropriately bin down to desired
    size.

    Parameters
    ----------
    datacube : PUNCHcube
        PUNCH data of interest
    glint_mask : np.array
        Boolean mask for masking out glint spheres
    """
    data_uncertainty = datacube.uncertainty.array
    data_only = datacube.data
    data_err = data_only*data_uncertainty
    err_mask = np.isfinite(data_only)*np.isfinite(data_err) #JK note: I still don't really know what this is

    mask = glint_mask*err_mask # JK note: (apparently) a mask to flag which data is good to use inversion or not
    binned_mask = np.clip(bindown(mask,bindown_shape),1,None)

    solver_data = np.array([(bindown(mask*data_only,bindown_shape)/binned_mask).T])
    solver_err = np.array([((bindown(mask*data_err**2,bindown_shape))**0.5/binned_mask).T])
    good_data_flags = np.array([(bindown(mask,bindown_shape) > 0).T])

    solver_err += 0.01*np.abs(solver_data)+np.nanmin(solver_data[good_data_flags])*0.25 # supplement the errors with 1% of the data values
    solver_err[good_data_flags==0] = np.max(solver_data[good_data_flags])
    solver_err[np.isfinite(solver_err)==0] = np.max(solver_data[good_data_flags]) # Some nans are still getting into the errors somehow. Grrr.

    return solver_data, solver_err, good_data_flags

def remove_nfi_stray_light(datacube: PUNCHCube,
                           bin_factor: int = 4,
                           fwd_mat_smooth_rad = 0.0,
                           sphere1_center:tuple = (540,790),
                           sphere2_center:tuple = (540,1210),
                           glint_sphere_radius:int = 375,
                           glint_bottom_cut:int = 250,
                           bindown_shape = [512,512],
                           solver_tol=1.0e-5,
                           sky_reg=0.1,
                           inst_reg=0.1,
                           stray_reg=1.0e-10):
    """
    Remove the dynamic NFI stray light from a given PUNCHCube image.
    """
    # Generate forward matrices (kernels)
    x_offsets, y_offsets, crota_radians = get_fwd_mat_inputs(datacube=datacube,bin_factor=bin_factor)
    data_size = (datacube.meta['NAXIS1'].value, datacube.meta['NAXIS2'].value)

    # Note: The inputs for "generate_nfi_fwdmats" are expected to be np.arrays to accomodate for 
    # processing multiple files at a time and generating foward matrices in one function
    # TODO: Modify to optimize processing one file at a time w/o use of np.arrays
    # TODO: OR fix above to append to arrays in the case of processing multiple frames at a time
    nframes = 1
    amats = generate_nfi_fwdmats(nframes,
                                 data_size,
                                 np.array([x_offsets]),
                                 np.array([y_offsets]),
                                 np.array([crota_radians]),
                                 bin_factor=bin_factor,smooth_rad=fwd_mat_smooth_rad)

    # Mask out glint spheres
    #TODO: Make mask optional
    glint_mask = generate_glint_mask(datacube.data.shape,sphere1_center,sphere2_center,glint_sphere_radius,glint_bottom_cut)

    # Stray light model
    solver_data, solver_err, good_data_flags = get_solver_inputs(datacube, glint_mask, 
                                                                 bindown_shape=bindown_shape)
    soln_sky, soln_ins, soln_stray, soln_dat = reconstruct_nfi_straylight(solver_data, solver_err, amats, 
                                                                          good_data_flags,
                                                                          solver_tol=solver_tol, 
                                                                          sky_reg=sky_reg, 
                                                                          inst_reg=inst_reg,
                                                                          stray_reg=stray_reg)
    

    # Upsample back up to 2k
    orig_shape = datacube.data.shape
    scale_x = orig_shape[0]/bindown_shape[0]
    scale_y = orig_shape[1]/bindown_shape[1]

    upscaled_stray_light_model = zoom(soln_stray[0].T,(scale_x,scale_y),order=0)
    subtracted_data = (datacube.data*glint_mask) - upscaled_stray_light_model # subtract straylight model from data with glint masked out

    datacube.data[:] = subtracted_data

    return datacube
