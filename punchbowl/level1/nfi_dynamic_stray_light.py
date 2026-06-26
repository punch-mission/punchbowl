import os
import copy
import importlib
from sys import path

import astropy.wcs
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning
from scipy.ndimage import gaussian_filter

plt.rcParams.update({'image.origin':'lower'})

file_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(file_dir)
module_path = os.path.join(src_dir,'nfi_modules')
path.append(module_path)

# native libraries
import nfi_modules.fwdmats
import nfi_modules.reconstruct

importlib.reload(nfi_modules.fwdmats)
importlib.reload(nfi_modules.reconstruct)

from nfi_modules.fwdmats import generate_nfi_fwdmats
from nfi_modules.reconstruct import reconstruct_nfi_straylight
from nfi_modules.util import bindown

# punchbowl libraries
from punchbowl.data.punchcube import PUNCHCube


def get_center(crval,cdelt,bin_factor:int):
    return (-crval/cdelt)/bin_factor

def get_fwd_mat_inputs(data: PUNCHCube,
                       bin_factor:int):
    data_wcs = data.wcs
    data_meta = data.meta

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

def get_solver_inputs(data_cube,
                      smask:np.array,
                      bindown_shape:list=[512,512]):
	data_uncertainty = data_cube.uncertainty.array
	data_only = data_cube.data
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

def remove_nfi_stray_light(data: PUNCHCube,
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
    
    xcens, ycens, crots = get_fwd_mat_inputs(data=data,bin_factor=bin_factor)
    data_size = [1, data.meta['NAXIS1'].value, data.meta['NAXIS2'].value]
    amats = generate_nfi_fwdmats(data_size,xcens,ycens,crots,bin_fac=bin_factor,smooth_rad=fwd_mat_smooth_rad)

    #TODO: Make mask optional (and/or mask generating lives outside this function)
    smask = glint_mask(data.data.shape,sc1,sc2,srad,bottom_cut)

    dsol, esol, gsol = get_solver_inputs(data, smask, bindown_shape=bindown_shape)
    soln_sky, soln_ins, soln_stray, soln_dat = reconstruct_nfi_straylight(dsol, esol, amats, gsol,
                                                                          solver_tol=solver_tol, 
                                                                          sky_reg=sky_reg, 
                                                                          inst_reg=inst_reg,
                                                                          stray_reg=stray_reg)
    
    return
