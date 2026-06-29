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


def get_center(crval,cdelt,bin_factor:int):
    return (-crval/cdelt)/bin_factor

def get_fwd_mat_inputs(datacube: PUNCHCube,
                       bin_factor:int):
    data_wcs = datacube.wcs
    data_meta = datacube.meta

    xcens = np.array([get_center(data_wcs.wcs.crval[0],data_wcs.wcs.cdelt[0],bin_factor)])
    ycens = np.array([get_center(data_wcs.wcs.crval[1],data_wcs.wcs.cdelt[1],bin_factor)])

    crots = np.array([data_meta['CROTA'].value*np.pi/180])

    return xcens, ycens, crots

def glint_mask(data_shape,
               sc1:tuple,
               sc2:tuple,
               srad:int,
               bottom_cut:int):
    xa, ya = np.indices(data_shape)
    mask =  ((((xa-sc1[0])**2+(ya-sc1[1])**2)**0.5 > srad)*
        ((((xa-sc2[0])**2+(ya-sc2[1])**2)**0.5 > srad))*
        (xa>bottom_cut))

    return mask

def get_solver_inputs(datacube,
                      smask: np.ndarray,
                      bindown_shape=(512, 52)):
    data_uncertainty = datacube.uncertainty.array
    data_only = datacube.data
    data_err = data_only*data_uncertainty
    msk = np.isfinite(data_only)*np.isfinite(data_err)

    mask = smask*msk
    nmask = np.clip(bindown(mask,bindown_shape),1,None)

    dsol = np.array([(bindown(mask*data_only,bindown_shape)/nmask).T])
    esol = np.array([((bindown(mask*data_err**2,bindown_shape))**0.5/nmask).T])
    gsol = np.array([(bindown(mask,bindown_shape) > 0).T])

    esol += 0.01*np.abs(dsol)+np.nanmin(dsol[gsol])*0.25 # supplement the errors with 1% of the data values
    esol[gsol==0] = np.max(dsol[gsol])
    esol[np.isfinite(esol)==0] = np.max(dsol[gsol]) # Some nans are still getting into the errors somehow. Grrr.

    return dsol, esol, gsol

def remove_nfi_stray_light(datacube: PUNCHCube,
                           bin_factor: int = 4,
                           fwd_mat_smooth_rad = 0.0,
                           sc1:tuple = (540,790),
                           sc2:tuple = (540,1210),
                           srad:int = 375,
                           bottom_cut = 250,
                           bindown_shape = [512,512],
                           solver_tol=1.0e-5,
                           sky_reg=0.1,
                           inst_reg=0.1,
                           stray_reg=1.0e-10):
    
    # Generate forward matrices (kernels)
    xcens, ycens, crots = get_fwd_mat_inputs(datacube=datacube,bin_factor=bin_factor)
    data_size = [1, datacube.meta['NAXIS1'].value, datacube.meta['NAXIS2'].value]
    amats = generate_nfi_fwdmats(data_size,xcens,ycens,crots,bin_fac=bin_factor,smooth_rad=fwd_mat_smooth_rad)

    # Mask out glint spheres
    #TODO: Make mask optional (and/or mask generating lives outside this function)
    smask = glint_mask(datacube.data.shape,sc1,sc2,srad,bottom_cut)

    # Stray light model
    dsol, esol, gsol = get_solver_inputs(datacube, smask, bindown_shape=bindown_shape)
    soln_sky, soln_ins, soln_stray, soln_dat = reconstruct_nfi_straylight(dsol, esol, amats, gsol,
                                                                          solver_tol=solver_tol, 
                                                                          sky_reg=sky_reg, 
                                                                          inst_reg=inst_reg,
                                                                          stray_reg=stray_reg)
    
    
    subtracted_data = (dsol[0] - soln_stray[0]).T

    # Upsample final data back up to 2k
    orig_shape = datacube.data.shape
    scale_x = orig_shape[0]/bindown_shape[0]
    scale_y = orig_shape[1]/bindown_shape[1]

    final_subtracted = zoom(subtracted_data,(scale_x,scale_y),order=0)

    #TODO: Save to final to punch ndcube?

    return final_subtracted
