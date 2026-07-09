"""
(Joe's Notes)
These modules compute and assemble the forward matrices for the stray light problem.
generate_nfi_fwdmats is called in the main correction script before the reconstruction
(by `reconstruct_nfi_straylight`, which uses the output of `generate_nfi_fwdmats` as input)
`generate_nfi_fwdmats` has the following inputs:
datsiz: 
	the size of the data cube (number of frames, nx, and ny). 
	
	The code uses the convention that the initial spatial axis is `x` and the subsequent spatial axis is `y`, 
	so that, for example, `NAXIS1` of a fits file corresponds to `nx` and `NAXIS2` corresponds to `ny`. 
	The data arrays returned by the astropy fits routines use the opposite convention (e.g., the initial 
	axis of the data array corresponds to `NAXIS2`, which is usually `y`, for a 2 axis image) so the data 
	arrays need to be transposed prior to being input to the reconstruct routine.
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


def generate_nfi_fwdmats(nframes:int ,data_size: tuple, x_offsets: np.array, y_offsets: np.array, crots: np.array, 
						 bin_factor: int = 4, smooth_rad: float = 0.05, radial_size: float = 175.4, 
						 elon_abs=130, cx=1009, cy=1029, nstray=None):
	"""
	Creates the forward matrices used to create the dynamic straylight models in NFI.

	This function is called in the main correction script before the reconstruction
	(by `reconstruct_nfi_straylight`, which uses the output of `generate_nfi_fwdmats` as input)

	Parameters
	----------
	nframes : int
		number of frames
	data_size : tuple
		The size of the data cube (nx, and ny). 
		
		The code uses the convention that the initial spatial axis is `x` and the subsequent spatial axis is `y`, 
		so that, for example, `NAXIS1` of a fits file corresponds to `nx` and `NAXIS2` corresponds to `ny`. 
		The data arrays returned by the astropy fits routines use the opposite convention (e.g., the initial 
		axis of the data array corresponds to `NAXIS2`, which is usually `y`, for a 2 axis image) so the data 
		arrays need to be transposed prior to being input to the reconstruct routine.
	x_offsets : np.array
		The x offsets of each frame in pixels. 
		For a correctly aligned fits file, these should be CRVAL1/CDELT1
	y_offsets : np.array
		The y offsets of each frame in pixels. 
		For a correctly aligned fits file, these should be CRVAL2/CDELT2
	crots : np.array
		The rotation of each frame relative to the fixed sky, about its center. 
	bin_factor : int
		How much to bin down the data for speed (default 4). 
		Needs to be set the same in `reconstruct_nfi_straylight`.
	smooth_rad : float
		If > 0, smooth the stray light kernels azimuthally with this radius (in radians)
	radial_size : float
	"""
	nx, ny = data_size
	dimensions = np.round(np.array([nx,ny])/bin_factor).astype(np.int32)
	source_sky = get_sky_source(dimensions)
	detector0 = get_detector(dimensions)
	source_inst = get_sky_source(dimensions)
	amat_inst = esr(source_inst, detector0, CoordTransform)

	detectors, amats_sky = [], []
	for i in range(nframes):
		detectors.append(get_detector(dimensions, center=[x_offsets[i], y_offsets[i]], crota=crots[i]))
		amats_sky.append(esr(source_sky, detectors[i], CoordTransform))

	im_size = dimensions[0]
	# Note we do not include the endpoint of the interval
	# since that would put duplicate kernels at 0 and 2*pi...
	if(nstray is None):
		nstray = im_size

	kernel_angles = 2*np.pi*np.arange(nstray)/nstray
	kernels = generate_kernels(kernel_angles,
						  radial_size=radial_size,
						  elon_abs=elon_abs,
						  image_size=im_size,
						  cx=cx/bin_factor,
						  cy=cy/bin_factor)
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
	for i in range(nframes):
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
	for i in range(nframe):
		amat_out += csc_resize(amats["stray"].T, nsrc, ndat, nsky+nins+i*im_size, i*npix).T

	return amat_out


def get_rotmat_2d(theta):
	"""
	Get 2d rotation matrix for a given theta angle.

	Parameter
	---------
	theta : float
		Rotation angle

	Returns
	-------
	rotmat : np.array
		Rotation matrix
	"""
	rotmat = np.array([[np.cos(theta),-np.sin(theta)],
					 [np.sin(theta),np.cos(theta)]])
	return rotmat 

def get_sky_source(dimensions:np.array, crota:float=0.0, center:np.array=np.array([0,0]), src_subgrid_fac=2):
	"""
	Get source grid for sky model.

	A source grid is the same as the ElementGrid except that the evaluation grid for
    the source basis functions is a subgrid consisting of indices rather than using a
    physically dimensioned coordinate system.

	Parameters
	----------
	dimensions : np.array
		The dimensions of the source grid, a 1-D array containing [nx0,nx1,...]
	crota : float, default = 0.0
		Image rotation angle
	center : np.array, default = np.array([0,0])

	src_subgrid_fac : int, default = 2

	Returns
	-------
	SourceGrid
		Source grid for sky model
		
	"""
	forward_transform = get_rotmat_2d(crota)
	src_frame = Trivialframe(["x", "y"]) # saves the coordinate names?
	origin = forward_transform.dot(-0.5*dimensions)+center

	src_coords = CoordGrid(dimensions, origin, forward_transform, src_frame)
	return SourceGrid(src_coords, None, bin_function, nsubgrid=src_subgrid_fac)

def get_detector(dimensions:np.array, crota:float = 0.0, center:np.array = np.array([0,0]), det_subgrid_fac:int = 3):
	"""
	Get detector grid.

	The detector grid is a straight implementation of the base class: "Element Grid"

	Parameters
	----------
	dimensions : np.array
		The dimensions of the detector grid, a 1-D array containing [nx0,nx1,...]
	crota : float, default = 0.0
		Image rotation angle
	center : np.array, default = np.array([0,0])

	det_subgrid_fac : int, default = 3

	Returns
	--------
	DetectorGrid
		A coordinate grid for the detector
	"""
	forward_transform = get_rotmat_2d(crota)
	det_frame = Trivialframe(["x", "y"])
	origin = forward_transform.dot(-0.5*dimensions+center)

	det_coords = CoordGrid(dimensions, origin, forward_transform, det_frame)
	psf_covariance = get_2d_covariance([0.5,0.5],0.0)
	inverse_psf_covariance = np.linalg.inv(psf_covariance)
	return DetectorGrid(det_coords, [inverse_psf_covariance], n_dimensional_gaussian_psf, nsubgrid=det_subgrid_fac, threshold=1.0e-3, footprint=[25, 25])

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
