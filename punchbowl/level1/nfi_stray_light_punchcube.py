# %%
# Should only need a standard stack of numpy, scipy, astropy. Sunpy may be beneficial?
import os, copy, importlib, numpy as np, matplotlib.pyplot as plt
from sys import path
from astropy.time import Time
from astropy.io import fits

from astropy.wcs import WCS, FITSFixedWarning
from scipy.ndimage import gaussian_filter
plt.rcParams.update({'image.origin':'lower'})

file_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(file_dir)
module_path = os.path.join(file_dir,'nfi_modules')
path.append(src_dir)
path.append(module_path) 

# %% native libraries
import nfi_modules.fwdmats, nfi_modules.reconstruct
importlib.reload(nfi_modules.fwdmats)
importlib.reload(nfi_modules.reconstruct)

from nfi_modules.reconstruct import reconstruct_nfi_straylight
from nfi_modules.fwdmats import generate_nfi_fwdmats
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

# %% LOAD AND SORT DATA BY TIME---------------------------------------------------------------
# Load data files and sort them by time:
datarr, errarr, namarr, hdrarr, timarr = [], [], [], [], []
for i in range(0,len(filenames[:10])):
	hdul = fits.open(os.path.join(punchdir,filenames[i]))
	#hdul.info()
	dat, relerr, hdr = hdul[1].data, 1.0/hdul[2].data, hdul[1].header
	bad_dat = np.nansum(dat)*1.0e6 >= 1000 + hdr['OUTLIER'] + hdr['BADPKTS'] #if (total num of nan)*1e6 >= 1000 + "Probable bad-image status" + ???
	if(bad_dat==0): # if false
		timarr.append(Time(hdr['DATE-OBS']).unix)
		datarr.append(dat)
		hdrarr.append(hdr)
		errarr.append(dat*relerr)
		namarr.append(filenames[i])
	hdul.close()
tsort = np.argsort(timarr)

datarr = np.array(datarr)[tsort]
errarr = np.array(errarr)[tsort]
timarr = np.array(timarr)[tsort]
namarr = np.array(namarr)[tsort]
mskarr = (np.isfinite(datarr)*np.isfinite(errarr))[tsort]

# %% MASK DATA WITH NANs----------------------------------------------------------------------
# Sanitize the data and errors so we don't get nan leakage:
mskarr = (np.isfinite(datarr)*np.isfinite(errarr))[tsort]

datarr[mskarr==False] = np.median(datarr[mskarr])
errarr[mskarr==False] = np.max(datarr[mskarr])

# %% COMPUTE "MINIMUM" (dmin)-----------------------------------------------------------------
# Compute a 'minimum' (actually bottom 5th percentile at each pixel) image
# which can be used to estimate the sphere and post pattern.
minlvl = round(0.05*len(datarr))
dmin = np.zeros(datarr.shape[1:])
for i in range(0,datarr.shape[1]):
    for j in range(0,datarr.shape[2]):
        if(np.sum(mskarr[:,i,j]) > 50):
            srtvals = np.sort(datarr[:,i,j][mskarr[:,i,j]])
            dmin[i,j] = srtvals[minlvl]


# %% RADIAL MINIMUM IMAGE---------------------------------------------------------------------
# This computes a radial minimum image which is an estimate of more diffuse stray light
# distinct from the sphere and post pattern. It can be subtracted from dmin in order to
# keep it from subtracting quite as much diffuse stray light. It's mostly an improvement
# but I think a more careful approach to the sphere and post pattern removal than this
# is needed.
xa, ya = np.indices(dmin.shape,dtype=float)
xa -= 0.5*dmin.shape[0]; ya -= 0.5*dmin.shape[1]
ra = np.sqrt(xa*xa+ya*ya)

nr_interp = 81
rad_interp = np.arange(nr_interp,dtype=float)*1024/(nr_interp-1)
radial_mins = np.zeros(len(rad_interp)-1)
for i in range(0,len(rad_interp)-1):
	radial_mins[i] = np.quantile(dmin[(ra <= rad_interp[i+1])*(ra > rad_interp[i])],0.0025)

radial_min_img = np.interp(ra, 0.5*(rad_interp[0:-1]+rad_interp[1:]), radial_mins)

# %% (JK added) Preview "radial_min_img"------------------------------------------------------
plt.imshow(radial_min_img)

# %% EXAMPLE DATA PLOT------------------------------------------------------------------------
fig=plt.figure(figsize=[16,10])
plt.imshow(datarr[5])

# %% PLOT: DMIN, RADIAL MIN IMG, DMIN - (minus) RADIAL MIN IMG--------------------------------
# Plotting the patterns
fig,axes=plt.subplots(nrows=1,ncols=3,figsize=[18,6])
axes[0].imshow(dmin**0.5,vmin=0,vmax=1.0e-9**0.5)
axes[0].set(title='Data minimum/quantile')
axes[1].imshow(radial_min_img**0.5,vmin=0,vmax=1.0e-9**0.5)
axes[1].set(title='Radial minimum image')
axes[2].imshow(np.clip(dmin-radial_min_img,0,None)**0.5,vmin=0,vmax=1.0e-9**0.5)
axes[2].set(title='Data minimum minus radial minimum')

# %% PLOT: EXAMPLE DATA , DATA - DMIN, DATA - DMIN + RADIAL MIN IMG----------------------------
# Plotting the patterns
fig,axes=plt.subplots(nrows=1,ncols=3,figsize=[18,6])
axes[0].imshow(np.clip(datarr[5],0,None)**0.5,vmin=0,vmax=1.0e-9**0.5)
axes[0].set(title='Example data')
axes[1].imshow(np.clip(datarr[5]-dmin,0,None)**0.5,vmin=0,vmax=1.0e-9**0.5)
axes[1].set(title='Example data minus data minimum image')
axes[2].imshow(np.clip(datarr[5]-dmin+radial_min_img,0,None)**0.5,vmin=0,vmax=1.0e-9**0.5)
axes[2].set(title='Data minimum minus data min plus radial min')

# %% CREATE INPUTS FOR FORWARD MATRIX GENERATOR------------------------------------------------
# Set up inputs to forward matrix generator. This uses just one index from the data array
# but multiple can also be used. That has been tested in the past but not very recently.
# if multiple are input (typically 3) it will create forward matrices for a detector-fixed
# per pixel pattern and a sky-fixed pattern as well as the diffuse stray light pattern
data_indices = [5]
datsz = [len(data_indices), hdrarr[0]['NAXIS1'], hdrarr[0]['NAXIS2']]
xcens = np.zeros(len(data_indices))
ycens = np.zeros(len(data_indices))
crots = np.zeros(len(data_indices))

# I had some difficulty getting the crvals in the headers to produce images that appeared
# to be aligned but that may be because I was recentering the crvals on the middle frame
# which can have odd effects when combined rotation. This doesn't appear to be happening here.
for i in range(0,len(data_indices)):
	xcens[i] = (-hdrarr[data_indices[i]]['CRVAL1']/hdrarr[data_indices[i]]['CDELT1'])/bin_fac
	ycens[i] = (-hdrarr[data_indices[i]]['CRVAL2']/hdrarr[data_indices[i]]['CDELT2'])/bin_fac
	crots[i] = hdrarr[data_indices[i]]['CROTA']*np.pi/180

# %%
import fwdmats
importlib.reload(fwdmats)
from fwdmats import generate_nfi_fwdmats, assemble_nfi_fwdmats

# %% FORWARD MATRICES DICTIONARY (amats)--------------------------------------------------------
# This generates a new sky-oriented and instrument oriented forward matrix each time
# which is inefficient. The instrment-oriented forward matrix only needs to be computed
# once and the sky oriented one only needs to be computed once for each data file.
amats = generate_nfi_fwdmats(datsz, xcens, ycens, crots, bin_fac=bin_fac, smooth_rad=0.0)

# %% AMATS KEYS-------------------------------------------------------------------------------
amats.keys()

# %% STRAY LIGHT FORWARD MATRIX (KERNEL)------------------------------------------------------
amats['stray']

# %% PLOT OF THE KERNEL (STRAY LIGHT FOWARD MATRIX)-------------------------------------------
# plot of the kernel
plt.imshow(amats['stray'][:,0].todense().A1.reshape([512,512]))

# %%
1.0/np.sum(amats['stray'][:,0] > 0)

# %%
fig=plt.figure(figsize=[20,20])
plt.imshow(amats['sky'][0][:,8192+128].todense().A1.reshape([512,512]))

# %% MASK OUT GLINT (fixed in position)--------------------------------------------------------------
# For now the plan is just to mask out the sphere pattern until we have a better way to prefilter it:
xa, ya = np.indices(datarr[5].shape)
sc1 = 540,790
sc2 = 540,1210
srad = 375

smask = ((((xa-sc1[0])**2+(ya-sc1[1])**2)**0.5 > srad)*
		((((xa-sc2[0])**2+(ya-sc2[1])**2)**0.5 > srad))*
		(xa>250))

plt.imshow(datarr[5]*smask)

# %% CREATE SOLVERS-----------------------------------------------------------------------------------
from scipy.sparse import diags

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
solns_sky, solns_stray, solns_dat = [],[],[]
for i in range(0,1):#10): # can apply to as many of the data as needed. Just using one here for testing.
	# We're binning down by a factor of 4 here. This makes the forward problem faster and more tractable
	# for the solver. If an element of the data contains no good pixels it is masked out, otherwise
	# the downbinning uses whatever pixels are available. To do the full 2k images, we can probably
	# use the binned down estimate for the stray light coefficients, just regenerating the stray light kernels
	# to 2k. May need to increase the number of stray light terms (nstray in generate_nfi_fwdmats). This
	# is not completely tested and had been locked to the number of pixels...
	data_index = data_indices[i]
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
	solns_sky.append(soln_sky)
	solns_stray.append(soln_stray)
	solns_dat.append(soln_dat)

# %% SKY SOLUTION-----------------------------------------------------------------------------
# The sky solution:
fig,axes = plt.subplots(nrows=1,ncols=2,figsize=[20,10])
#fig = plt.figure(figsize=[15,15])
axes[0].imshow(dsol[0].T**0.5,vmin=0,vmax=0.5e-9**0.5)
axes[1].imshow((solns_sky[0]).T**0.5,vmin=0,vmax=0.5e-9**0.5)

# %% DATA vs STRAY LIGHT SOLUTION (stray light model)-----------------------------------------
# The stray solution:
fig,axes = plt.subplots(nrows=1,ncols=2,figsize=[20,10])
fig = plt.figure(figsize=[15,5])
axes[0].imshow(dsol[0].T**0.5,vmin=0,vmax=0.5e-9**0.5)
axes[1].imshow((solns_stray[0][0]).T)

# %% STRAY LIGHT SOLUTION ONLY----------------------------------------------------------------
# The stray light solution:
fig = plt.figure(figsize=[15,15])
plt.imshow(np.clip(solns_stray[0][0],0,None).T**0.5,vmin=0,vmax=0.5e-9**0.5)

# %% SOLVED IMAGE (data - (minus) stray light model)------------------------------------------
fig = plt.figure(figsize=[15,15])
plt.imshow((dsol[0]-solns_stray[0][0]).T)

# %% VISUALIZING ALL THE PRODUCTS (dsol, esol, gsol; solns_sky, solns_stray, soln_dat)--------
fig,axes = plt.subplots(nrows=2,ncols=3,figsize=[20,10])
#fig = plt.figure(figsize=[15,15])
ax00 = axes[0][0].imshow(dsol[0].T**0.5,cmap='gray',vmin=0,vmax=0.5e-9**0.5)
axes[0][0].set_title('dsol: "data"')
ax01 = axes[0][1].imshow(esol[0].T**0.5,cmap='gray',vmin=0,vmax=0.5e-9**0.5)
axes[0][1].set_title('esol: error')
ax02 = axes[0][2].imshow(gsol[0].T**0.5,cmap='gray',vmin=0,vmax=0.5e-9**0.5)
axes[0][2].set_title('gsol: good_data')
fig.colorbar(ax00)
fig.colorbar(ax01)
fig.colorbar(ax02)
ax10 = axes[1][0].imshow((solns_sky[0]).T**0.5,cmap='gray',vmin=0,vmax=0.5e-9**0.5)
axes[1][0].set_title('solns_sky')
ax11 = axes[1][1].imshow((solns_stray[0][0]).T**0.5,cmap='gray',vmin=0,vmax=0.5e-9**0.5)
axes[1][1].set_title('solns_stray')
ax12 = axes[1][2].imshow((solns_dat[0]).T**0.5,cmap='gray',vmin=0,vmax=0.5e-9**0.5)
axes[1][2].set_title('solns_dat')
fig.colorbar(ax10)
fig.colorbar(ax11)
fig.colorbar(ax12)

# %% THREE PANEL FIGURE: DATA , STRAY LIGHT MODEL (SOLUTION) , SUBTRACTED IMAGE----------------
fig,axes = plt.subplots(nrows=1,ncols=3,figsize=[20,5])
axes[0].imshow(dsol[0].T**0.5,vmin=0,vmax=0.5e-9**0.5,cmap='gray')
axes[1].imshow((solns_stray[0][0]).T**0.5,vmin=0,vmax=0.5e-9**0.5,cmap='gray')
axes[2].imshow(((dsol[0]-solns_stray[0][0]).T)**0.5,vmin=0,vmax=0.5e-9**0.5,cmap='gray')
# %% Labelled, not scaled, and masked solved
fig,axes = plt.subplots(nrows=1,ncols=3,figsize=[20,5])
t = Time(timarr[data_indices[0]],format='unix')
t.format='iso'
fig.suptitle(t.value,fontsize=20)
sub_fntsz = 15
cmap_color = 'gray'

ax0 = axes[0].imshow(dsol[0].T,cmap=cmap_color)
axes[0].set_title('Data',fontsize=sub_fntsz)
ax1 = axes[1].imshow((solns_stray[0][0]).T,cmap=cmap_color)
axes[1].set_title('Stray Light Solution',fontsize=sub_fntsz)
ax2 =axes[2].imshow((((dsol[i]-solns_stray[i][0])*(nmask.T)).T),cmap=cmap_color)
axes[2].set_title('Subtracted (data - stray)',fontsize=sub_fntsz)

fig.colorbar(ax0)
fig.colorbar(ax1)
fig.colorbar(ax2)
# %%
