"""
These modules compute and assemble the forward matrices for the stray light problem.
generate_nfi_fwdmats is called in the main correction script before the reconstruction
(by reconstruct_nfi_straylight, which uses the output of generate_nfi_fwdmats as input)
generate_nfi_fwdmats has the following inputs:
datsiz: the size of the data cube (number of frames, nx, and ny). The code uses the convention that
			the initial spatial axis is x and the subsequent spatial axis is y, so that, for example,
			NAXIS1 of a fits file corresponds to nx and NAXIS2 corresponds to ny. The data arrays returned
			by the astropy fits routines use the opposite convention (e.g., the initial axis of the data array
			corresponds to NAXIS2, which is usually y, for a 2 axis image) so the data arrays need to be
			transposed prior to being input to the reconstruct routine.
xoffs: The x offsets of each frame in pixels. For a correctly aligned fits file, these should be
			CRVAL1/CDELT1
yoffs: The y offsets of each frame in pixels. For a correctly aligned fits file, these should be
			CRVAL2/CDELT2
crots: The rotation of each frame relative to the fixed sky, about its center.
bin_fac: How much to bin down the data for speed (default 4). Needs to be set the same in
			reconstruct_nfi_straylight.
smooth_rad: If > 0, smooth the stray light kernels azimuthally with this radius (in radians)
"""
import numpy as np
from element_functions import bin_function, get_2d_covariance, n_dimensional_gaussian_psf
from element_grid import DetectorGrid, SourceGrid
from element_source_responses import element_source_responses as esr
from indexgrid import CoordGrid
from scipy.sparse import csc_matrix, csr_matrix, diags
from straylight_kernels import generate_kernels, kernel_smoothing_matrix
from transforms import CoordTransform, Trivialframe


def generate_nfi_fwdmats(datsiz, xoffs, yoffs, crots, bin_fac=4, smooth_rad=0.05, radial_size=175.4, elon_abs=130, cx=1009, cy=1029, nstray=None):
	nframe, nx0, ny0 = datsiz
	dims = np.round(np.array([nx0,ny0])/bin_fac).astype(np.int32)
	source_sky = get_sky_source(dims)
	detector0 = get_detector(dims)
	source_inst = get_sky_source(dims)
	amat_inst = esr(source_inst, detector0, CoordTransform)

	detectors, amats_sky = [], []
	for i in range(nframe):
		detectors.append(get_detector(dims, center=[xoffs[i], yoffs[i]], crota=crots[i]))
		amats_sky.append(esr(source_sky, detectors[i], CoordTransform))

	im_size = dims[0]
	# Note we do not include the endpoint of the interval
	# since that would put duplicate kernels at 0 and 2*pi...
	if(nstray is None):
		nstray = im_size

	kernel_angles = 2*np.pi*np.arange(nstray)/nstray
	kernels = generate_kernels(kernel_angles,
						  radial_size=radial_size,
						  elon_abs=elon_abs,
						  image_size=im_size,
						  cx=cx/bin_fac,
						  cy=cy/bin_fac)
	k2 = np.array(kernels)
	for i in range(len(k2)):
		k2[i] = k2[i].T

	k2 = k2.reshape([nstray,im_size*im_size])
	smat = csc_matrix(kernel_smoothing_matrix(kernel_angles/2/np.pi))
	amat_stray = csr_matrix(k2.T)
	if(smooth_rad > 0):
		amat_stray = amat_stray*csc_matrix(kernel_smoothing_matrix(kernel_angles/2/np.pi, smooth_rad=smooth_rad))

	norms_inst = np.sum(amat_inst,axis=0).A1
	norms_inst = np.clip(norms_inst,0.05*np.mean(norms_inst),None)
	amat_inst = amat_inst*diags(1.0/norms_inst)

	norms_stray = np.sum(amat_stray,axis=0).A1
	norms_stray = np.clip(norms_stray,0.05*np.mean(norms_stray),None)
	amat_stray = amat_stray*diags(1.0/norms_stray)

	norms_sky = []
	for i in range(nframe):
		norms_sky.append(np.sum(amats_sky[i],axis=0).A1)
		norms_sky[i] = np.clip(norms_sky[i],0.05*np.mean(norms_sky[i]),None)
		amats_sky[i] = amats_sky[i]*diags(1.0/norms_sky[i])

	return {"inst":amat_inst, "sky":amats_sky, "stray":amat_stray, "im_size":im_size,
			"norms_inst":norms_inst, "norms_stray":norms_stray, "norms_sky":norms_sky}

def generate_stray_fwdmats(datsiz, xoffs, yoffs, crots, bin_fac=4, smooth_rad=0.05):
	nframe, nx0, ny0 = datsiz
	dims = np.round(np.array([nx0,ny0])/bin_fac).astype(np.int32)

	source_sky = get_sky_source(dims)#, origin, 0.0)
	source_inst = get_sky_source(dims)#, origin, 0.0)
	detector0 = get_detector(dims)#, origin, 0.0)

	xoffs -= np.median(xoffs); yoffs -= np.median(yoffs)

	im_size = dims[0]
	# Note we do not include the endpoint of the interval
	# since that would put duplicate kernels at 0 and 2*pi...
	kernel_angles = 2*np.pi*np.arange(im_size)/im_size
	kernels = generate_kernels(kernel_angles, radial_size=175.4, elon_abs=130, image_size=im_size)
	k2 = np.array(kernels)
	for i in range(len(k2)):
		k2[i] = k2[i].T
	k2 = k2.reshape([im_size,im_size*im_size])
	smat = csc_matrix(kernel_smoothing_matrix(kernel_angles/2/np.pi))
	amat_stray = csr_matrix(k2.T)
	if(smooth_rad > 0):
		amat_stray = amat_stray*csc_matrix(kernel_smoothing_matrix(kernel_angles/2/np.pi, smooth_rad=smooth_rad))

	return {"stray":amat_stray}

def assemble_nfi_fwdmats(amats):
	nframe = len(amats["sky"])
	npix = amats["inst"].shape[0]
	ndat = nframe*npix
	nsky = npix; im_size=amats["im_size"]
	nins=npix*(nframe > 1)
	nstr = nframe*amats["stray"].shape[1]
	nsrc = nsky+nins+nstr

	amat_out = csc_matrix(([],[],np.zeros(nsrc+1)), shape=(ndat,nsrc))

	for i in range(nframe):
		if(nframe == 1): 
			amat_out += csc_resize(amats["inst"], ndat, nsrc, i*npix, 0)
		else: 
			amat_out += csc_resize(amats["sky"][i], ndat, nsrc, i*npix, 0)

	for i in range(nframe):
		if(nframe > 1): 
			amat_out += csc_resize(amats["inst"], ndat, nsrc, i*npix, nsky)
		#amat_out += csc_resize(csc_matrix(diags(np.ones(nins))), ndat, nsrc, i*npix, nsky)
	for i in range(nframe):
		amat_out += csc_resize(amats["stray"].T, nsrc, ndat, nsky+nins+i*im_size, i*npix).T

	return amat_out


def get_rotmat_2d(theta):
	vec0 = np.array([[np.cos(theta),-np.sin(theta)],
					 [np.sin(theta),np.cos(theta)]])
	return vec0 # np.outer(vec0,vec0)+np.outer(vec1,vec1)

def get_sky_source(dims, crota=0.0, center=np.array([0,0]), scale=[1.0,1.0], src_subgrid_fac=2):
	fwdtransform = get_rotmat_2d(crota)
	src_frame = Trivialframe(["x", "y"])
	origin = fwdtransform.dot(-0.5*dims)+center
	src_coords = CoordGrid(dims, origin, fwdtransform, src_frame)
	return SourceGrid(src_coords, None, bin_function, nsubgrid=src_subgrid_fac)

def get_detector(dims, crota=0.0, center=np.array([0,0]), scale=[1.0, 1.0], det_subgrid_fac=3):
	det_frame = Trivialframe(["x", "y"])
	fwdtransform = get_rotmat_2d(crota)
	origin = fwdtransform.dot(-0.5*dims+center)
	det_coords = CoordGrid(dims, origin, fwdtransform, det_frame)
	psfcov = get_2d_covariance([0.5,0.5],0.0)
	ipsfcov = np.linalg.inv(psfcov)
	return DetectorGrid(det_coords, [ipsfcov], n_dimensional_gaussian_psf, nsubgrid=det_subgrid_fac, thold=1.0e-3, footprint=[25, 25])

def csc_resize(csc, rsiz, csiz, r0, c0):
	"""
	csc_matrix((data, indices, indptr), [shape=(M, N)])
		is the standard CSC representation where the row indices for column i
		are stored in indices[indptr[i]:indptr[i+1]] and their corresponding
		values are stored in data[indptr[i]:indptr[i+1]]. If the shape parameter
		is not supplied, the matrix dimensions are inferred from the index arrays.
	"""
	rinds = csc.indices+r0
	cinds = csc.indptr
	if c0 > 0:
		cinds = np.hstack([np.zeros(c0,dtype=np.int64),cinds])
	n_extra = (csiz-c0-csc.shape[1])
	if n_extra > 0:
		cinds = np.hstack([cinds,csc.indptr[-1]*np.ones(n_extra,dtype=np.int64)])
	print(rsiz, csiz, r0, c0, n_extra, csc.shape, cinds.shape)
	return csc_matrix((csc.data, rinds, cinds), shape=(rsiz,csiz))
