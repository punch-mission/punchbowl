# reconstruct_nfi_straylight is the routine that does the actual inversion of the sky and stray light components
# It has the following inputs:
# data: The images to invert, dimensions n_img, nx, ny
# errs: Uncertainties corresponding to the images
# amats: dictionary containing the forward matrices for the sky, (per-pixel) instrument, and stray light
#		sources. Created by fwdmats.generate_nfi_fwdmats
# good_dat: Array flagging which data are good to use in the inversion, same shape as data
# bin_fac: how much to bin down the data for speed (default: 4). fwdmats.generate_nfi_fwdmats
# 			must be called with the same bin_fac.
# errfac_systematic: An additional uncertainty of this factor multiplied by the data is added to the errors.
# solver_tol: Tolerance for the solver, default 2.5e-5
# sky_reg: Regularization factor for the sky source, larger values are a heavier penalty; default 1
# inst_reg: Regularization factor for per-pixel instrument source, default 1
# stray_reg: Regularization factor for the disk stray light functions, default 0.001
# mask_source: Attempts to mask off source coefficients with no connection to valid data. Not working.
import numpy as np
from fwdmats import assemble_nfi_fwdmats
from scipy.sparse import csc_matrix, csr_matrix, diags
from solver import sparse_nlmap_solver
from nfi_modules.util import bindown

def reconstruct_nfi_straylight(data, errs, amats, good_dat, bin_fac=4, errfac_systematic=0.01, mask_source=False,
					solver_tol=2.5e-5, sky_reg=1, inst_reg=1, stray_reg=0.001, datanorm=1.0e-10):
	
	from nfi_modules.util import bindown
	from scipy.ndimage import gaussian_filter
	from solver import sparse_nlmap_solver
	fwdmat = assemble_nfi_fwdmats(amats)

	nframe, nx, ny = data.shape[0], round(data.shape[1]), round(data.shape[2])
	im_size = nx; npix = nx*ny
	dims = np.array([nx,ny],dtype=np.int32)
	nstr_coeffs = amats['stray'].shape[1]
	nsky=npix; nins=npix*(nframe>1); nstr=nframe*nstr_coeffs
	nsrc = nsky+nins+nstr

	dat_bin = [d/datanorm for d in data]
	err_bin = [e/datanorm for e in errs]
	msk_bin = [g for g in good_dat]
	good_data = np.hstack([m.flatten() for m in msk_bin])

	src_maskmat = mask_sources(fwdmat)
	dat_maskmat = mask_data(fwdmat, good_data)
	print('src_maskmat',src_maskmat.shape,'dat_maskmat',dat_maskmat.shape)

	regvec = np.ones(fwdmat.shape[1])
	regvec[0:nsky] = sky_reg
	if(nins > 0): regvec[nsky:nsky+nins] *= inst_reg
	regvec[nsky+nins:] *= stray_reg
	regmat = diags(regvec)
	
	fwdmat_masked = dat_maskmat*fwdmat*src_maskmat.T
	regmat_masked = src_maskmat*regmat*src_maskmat.T

	flatdat = np.hstack([d.flatten() for d in dat_bin])
	flaterr = np.hstack([e.flatten() for e in err_bin]) + errfac_systematic*np.abs(flatdat)
	print(dat_maskmat.shape, flatdat.shape, flaterr.shape, fwdmat_masked.shape, regmat_masked.shape, src_maskmat.shape)
	solution = sparse_nlmap_solver(dat_maskmat*flatdat, dat_maskmat*flaterr, fwdmat_masked, adapt_lam=False,
								   reg_fac=1, dtype='float32', niter=40, solver_tol=solver_tol, sqrmap=False,
								   flatguess=True, silent=False, regmat=regmat_masked)#, guess = np.ones(reg_guess.size))

	soln = src_maskmat.T*solution[0]
	if(nins > 0):
		soln_sky = datanorm*(amats['sky'][0]*(soln[0:npix])).reshape(dims)
		soln_ins = datanorm*(amats['inst']*(soln[npix:2*npix])).reshape(dims)
	else:
		soln_sky = datanorm*(amats['inst']*(soln[0:npix])).reshape(dims)		
		soln_ins = np.zeros(dims)
	soln_stray = []
	for i in range(0,nframe):
		soln_stray.append(datanorm*(amats['stray']*(soln[nsky+nins+i*nstr_coeffs:nsky+nins+(i+1)*nstr_coeffs])).reshape(dims))

	return soln_sky, soln_ins, soln_stray, np.array(dat_bin)*datanorm


# Mask sources that aren't present in the data
def mask_sources(amat_in, mask_lvl=None):
	from scipy.sparse import csc_matrix
	dsums = np.sum(amat_in,axis=0).A1
	if(mask_lvl is None): mask_lvl = 0.05*np.mean(dsums)
	print('dsums:',dsums.shape,'mask_lvl:', mask_lvl.shape)
	smask = dsums >= mask_lvl
	nsrc_in = len(smask); nsrc_out = np.sum(smask)
	print(nsrc_in, nsrc_out)
	maskinds = np.arange(nsrc_in, dtype=np.uint64)
	input_maskinds = maskinds[smask]
	output_maskinds = np.arange(nsrc_out,dtype=np.uint64)
	maskmat_vals = np.ones(nsrc_out,dtype=np.float32)
	maskmat = csc_matrix((maskmat_vals,(output_maskinds, input_maskinds)),shape=[nsrc_out, nsrc_in])
	return maskmat

# Mask data that aren't connected to the sources (or to anything)
def mask_data(amat_in, dmask_in, mask_lvl=None):
	from scipy.sparse import csc_matrix
	dsums = np.sum(amat_in,axis=1).A1
	if(mask_lvl is None): mask_lvl = 0.05*np.mean(dsums)
	dmask = dmask_in*(dsums >= mask_lvl)
	ndat_in = len(dmask); ndat_out = np.sum(dmask)
	maskinds = np.arange(ndat_in, dtype=np.uint64)
	input_maskinds = maskinds[dmask]
	output_maskinds = np.arange(ndat_out,dtype=np.uint64)
	maskmat_vals = np.ones(ndat_out,dtype=np.float32)
	maskmat = csc_matrix((maskmat_vals,(output_maskinds, input_maskinds)),shape=[ndat_out,ndat_in])
	return maskmat
