# %%
# Should only need a standard stack of numpy, scipy, astropy. Sunpy may be beneficial?
import os
import copy
import importlib
from sys import path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning
from scipy.ndimage import gaussian_filter

plt.rcParams.update({'image.origin':'lower'})

file_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(file_dir)
module_path = os.path.join(file_dir,'nfi_modules')
path.append(src_dir)
path.append(module_path)

# %% native libraries
import nfi_modules.fwdmats
import nfi_modules.reconstruct

importlib.reload(nfi_modules.fwdmats)
importlib.reload(nfi_modules.reconstruct)

from nfi_modules.fwdmats import generate_nfi_fwdmats
from nfi_modules.reconstruct import reconstruct_nfi_straylight
from nfi_modules.util import bindown

#punch modules
import punchbowl
from punchbowl.data import punch_io, visualize

# %% [markdown]
# Data obtaining instructions from Sam:
# If you want to start exploring now, I think our X files are the right ones to use. These are partway between L0 and L1. They're in MSB units, with the NFI speckle pattern corrected, cosmic rays and CCD spikes are removed. The FITS files are compressed, so HDU 0 is empty, the image is in HDU 1, and the uncertainty layer is in HDU 2. Values of inf in the uncertainty layer mark pixels whose values were filled in during the de-streaking or de-spiking

# You'll want XR4 files. R marks a cleaR image, and 4 is NFI

# The full archive is at https://umbra.nascom.nasa.gov/punch/1/XR4/

# This wget command will download all the fits files for a given date wget -r -l1 --no-parent --no-directories -A "PUNCH_L1_XR4_20251001*_v0j.fits" -R "*.html*,index*,*tmp*" https://umbra.nascom.nasa.gov/punch/1/XR4/2025/10/01/

# The uncertainty layer is saved in the FITS file as relative reciprocal uncertainty, for reasons. If you pip install punchbowl, you can use punchbowl.data.load_ndcube_from_fits to read a PUNCH file and do everything "right", including unpacking the uncertainty. It returns an NDCube---if you haven't used those, the data array is at mycube.data, the uncertainty at mycube.uncertainty.array, the wcs at mycube.wcs, and the metadata from the FITS header at mycube.meta. The meta is a PUNCH-specific NormalizedMetadata object. It acts like a dict of header keys and values, except that to access a value you have to do e.g. mycube.meta['CRPIX1'].value

# %% SAVE DATA FILENAMES----------------------------------------------------------------------
#punch190
punchdir = '/mnt/archive/soc/data/1/XR4/2025/10/01'
# #JK local PC
# punchdir = '/home/jasminekobayashi/data/punch/XR4-2025-10-01/01/'
# #JK loaner wsl
# punchdir = '/home/jkobayashi/data/punch/XR4-2025-10-01/01/'

filenames = []
for file in os.listdir(punchdir):
	if file.endswith('_v0j.fits'):
		filenames.append(file)

bin_fac = 4 # For processing we bin down the data by this factor

# %%
test_file = filenames[0]
data_cube = punch_io.load_ndcube_from_fits(punchdir+'/'+test_file)

# %% CREATE INPUTS FOR FORWARD MATRIX GENERATOR------------------------------------------------
def get_center(crval,cdelt,bin_factor):
    return (-crval/cdelt)/bin_factor

data_wcs = data_cube.wcs
data_meta = data_cube.meta

xcens = np.array([get_center(data_wcs.wcs.crval[0],data_wcs.wcs.cdelt[0],bin_fac)])
ycens = np.array([get_center(data_wcs.wcs.crval[1],data_wcs.wcs.cdelt[1],bin_fac)])

datsz = [1, data_meta['NAXIS1'].value, data_meta['NAXIS2'].value]
crots = np.array([data_meta['CROTA'].value*np.pi/180])

# %% FORWARD MATRICES DICTIONARY (amats)--------------------------------------------------------
# This generates a new sky-oriented and instrument oriented forward matrix each time
# which is inefficient. The instrment-oriented forward matrix only needs to be computed
# once and the sky oriented one only needs to be computed once for each data file.
amats = generate_nfi_fwdmats(datsz, xcens, ycens, crots, bin_fac=bin_fac, smooth_rad=0.0)

# %% AMATS KEYS-------------------------------------------------------------------------------
amats.keys()

# %% STRAY LIGHT FORWARD MATRIX (KERNEL)------------------------------------------------------
amats['stray']

# %% PLOT OF THE KERNEL (STRAY LIGHT FORWARD MATRIX)-------------------------------------------
# plot of the kernel
# plt.imshow(amats['stray'][:,0].todense().A1.reshape([512,512]))

# %% MASK OUT GLINT (fixed in position)--------------------------------------------------------------
# For now the plan is just to mask out the sphere pattern until we have a better way to prefilter it:
sc1 = 540,790
sc2 = 540,1210
srad = 375
bottom_cut = 250

def glint_mask(data_shape,sc1,sc2,srad,bottom_cut):
	xa, ya = np.indices(data_shape)
	mask =  ((((xa-sc1[0])**2+(ya-sc1[1])**2)**0.5 > srad)*
		((((xa-sc2[0])**2+(ya-sc2[1])**2)**0.5 > srad))*
		(xa>bottom_cut))

	return mask

smask = glint_mask(data_cube.data.shape,sc1,sc2,srad,bottom_cut)

# plt.imshow(data_cube.data*smask)

# %% CREATE SOLVERS-----------------------------------------------------------------------------------
# This does the single-frame PSF correction. Multi-frame capability should still be present
# but will require massaging the inputs.
# Inputs to the reconstructor/solver:
# solver_tol: internal tolerance for the gmres solver (scipy.sparse.lgmres). I've been using 1e-5.
#		1e-6 seemed to work but slower and maybe more issues with numerical precision. 1e-4 untested,
#		1e-3 almost certainly not enough precision.
# sky_reg: Regularization factor for the per-pixel sky component, of order one to ten seems to work.
#		generally this should be set to obtain chi squared of order unity and/or to make chi squared
#		higher of order 25% compared to if it's very small. Larger values will be more conservative
#		about what is assigned to the sky component.
# inst_reg: Regularization factor for per-pixel instrument component similar to sky-reg. Not used
#		for single frame.
# stray_reg: Regularization for stray light. This can be very small since the stray light generally
#		should be limited by the small number of corfficients.

# We're binning down by a factor of 4 here. This makes the forward problem faster and more tractable
	# for the solver. If an element of the data contains no good pixels it is masked out, otherwise
	# the downbinning uses whatever pixels are available. To do the full 2k images, we can probably
	# use the binned down estimate for the stray light coefficients, just regenerating the stray light kernels
	# to 2k. May need to increase the number of stray light terms (nstray in generate_nfi_fwdmats). This
	# is not completely tested and had been locked to the number of pixels...
	
mask = smask*mskarr[data_index]
nmask = np.clip(bindown(mask,[512,512]),1,None)
dsol = np.array([(bindown(mask*(datarr[data_index]-0.0*(dmin-radial_min_img)),[512,512])/nmask).T])
esol = np.array([((bindown(mask*errarr[data_index]**2,[512,512]))**0.5/nmask).T])
gsol = np.array([(bindown(mask,[512,512]) > 0).T])
esol += 0.01*np.abs(dsol)+np.nanmin(dsol[gsol])*0.25 # supplement the errors with 1% of the data values
esol[gsol==0] = np.max(dsol[gsol])
esol[np.isfinite(esol)==0] = np.max(dsol[gsol]) # Some nans are still getting into the errors somehow. Grrr.

soln_sky, soln_ins, soln_stray, soln_dat = reconstruct_nfi_straylight(dsol, esol, amats, gsol,
																		solver_tol=1.0e-5, sky_reg=0.1, inst_reg=0.1,
																		stray_reg=1.0e-10)

