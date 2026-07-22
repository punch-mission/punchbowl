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
from scipy.sparse import csc_matrix, csr_matrix, diags

from punchbowl.level1.nfi_modules.element_functions import (
    bin_function_evaluator,
    get_2d_covariance,
    n_dimensional_gaussian_psf,
)
from punchbowl.level1.nfi_modules.element_grid import DetectorGrid, SourceGrid
from punchbowl.level1.nfi_modules.element_source_responses import element_source_responses as esr
from punchbowl.level1.nfi_modules.indexgrid import CoordGrid
from punchbowl.level1.nfi_modules.straylight_kernels import generate_kernels, kernel_smoothing_matrix
from punchbowl.level1.nfi_modules.transforms import CoordTransform, Trivialframe


def generate_nfi_forward_matrices(
    nframes: int,
    data_size: tuple,
    x_offsets: np.ndarray,
    y_offsets: np.ndarray,
    crots: np.ndarray,
    bin_factor: int = 4,
    smooth_rad: float = 0.05,
    radial_size: float = 175.4,
    elon_abs=130,
    cx=1009,
    cy=1029,
    nstray=None,
    thread_count: int = 5,
):
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

    Returns
    -------
    forward_matrices: dictionary of ndarray objects
            Dictionary object containing all forward matrices and normalizers(?) of each of the following: the instrument (`inst`),
            the sky model (`sky`), and the dynamic stray light model (`stray`); as well as image size.

            Where `inst`, `sky`, and `stray` as keywords return the forward matrix for each relevant context, and `norms_inst`,
            `norms_sky`, and `norms_stray` gives the normalizers used to make each of the respective forward matrices.
            Keyword, `im_size` give the image size
    """
    nx, ny = data_size
    dimensions = np.round(np.array([nx, ny]) / bin_factor).astype(np.int32)
    source_sky = get_sky_source(dimensions)
    detector0 = get_detector(dimensions)
    source_inst = get_sky_source(dimensions)
    inst_forward_mat = esr(source_inst, detector0, CoordTransform)

    detectors = []
    sky_forward_mat = []
    for i in range(nframes):
        detectors.append(get_detector(dimensions, center=[x_offsets[i], y_offsets[i]], crota=crots[i]))
        sky_forward_mat.append(esr(source_sky, detectors[i], CoordTransform))

    im_size = dimensions[0]
    # Note we do not include the endpoint of the interval
    # since that would put duplicate kernels at 0 and 2*pi...
    if nstray is None:
        nstray = im_size

    kernel_angles = 2 * np.pi * np.arange(nstray) / nstray
    kernels = generate_kernels(
        kernel_angles,
        radial_size=radial_size,
        elon_abs=elon_abs,
        image_size=im_size,
        cx=cx / bin_factor,
        cy=cy / bin_factor,
        n_threads=thread_count,
    )
    for i in range(len(kernels)):
        kernels[i] = kernels[i].T

    kernels = kernels.reshape([nstray, im_size * im_size])
    # Note: csr-matrix = "compressed sparse row matrix"
    stray_forward_mat = csr_matrix(kernels.T)
    if smooth_rad > 0:
        # Note: csc-matrix = "compressed sparse column matrix"
        stray_forward_mat = stray_forward_mat * csc_matrix(
            kernel_smoothing_matrix(kernel_angles / 2 / np.pi, smooth_rad=smooth_rad)
        )

    # Create Forward Matrix for Instrument
    norms_inst = np.sum(inst_forward_mat, axis=0).A1
    norms_inst = np.clip(norms_inst, 0.05 * np.mean(norms_inst), None)
    inst_forward_mat = inst_forward_mat * diags(1 / norms_inst)

    # Create Forward Matrix for Stray Light model
    norms_stray = np.sum(stray_forward_mat, axis=0).A1
    norms_stray = np.clip(norms_stray, 0.05 * np.mean(norms_stray), None)
    stray_forward_mat = stray_forward_mat * diags(1 / norms_stray)

    # Create Forward Matrix Sky model (includes background stars and f-corona)
    norms_sky = []
    for i in range(nframes):
        norms_sky.append(np.sum(sky_forward_mat[i], axis=0).A1)
        norms_sky[i] = np.clip(norms_sky[i], 0.05 * np.mean(norms_sky[i]), None)
        sky_forward_mat[i] = sky_forward_mat[i] * diags(1 / norms_sky[i])

    return {
        "inst": inst_forward_mat,
        "sky": sky_forward_mat,
        "stray": stray_forward_mat,
        "im_size": im_size,
        "norms_inst": norms_inst,
        "norms_stray": norms_stray,
        "norms_sky": norms_sky,
    }


def assemble_nfi_fwdmats(fwdmats_dict):
    """
    Assemble a single sparse design matrix from per-component system matrices.
    Used by `reconstruct.reconstruct_nfi_straylight`

    Parameters:
    -----------
    fwdmats_dict: dict
            dictionary containing the forward matrices for the sky, (per-pixel) instrument, and stray light
            sources.
            Created by `fwdmats.generate_nfi_forward_matrices`
    Returns:
    --------
    fwdmat_final: scipy.sparse.csc_matrix
            The single sparse design matrix made with totaled values from the per-component system matrices as appropriate.

            The final object is a compressed sparse column matrix object with shape `(n_data_points, n_source)`, where `n_data_points` the number of
            forward matrices of the `sky` component of `fwdmats_dict` (i.e. number of frames) multiplied by the number of detector pixels
            per frame; and `n_source` is the total number of parameters from the sky model, instrument, and stray light model.
    """
    # TODO: (JK) not totally confident about this docstring explanation.
    n_frames = len(fwdmats_dict["sky"])
    n_pixels = fwdmats_dict["inst"].shape[0]
    n_data_points = n_frames * n_pixels

    im_size = fwdmats_dict["im_size"]
    n_sky = n_pixels  # number of sky parameters (one per pixel)
    n_inst = n_pixels * (n_frames > 1)  # number of instrument parameters
    n_stray = n_frames * fwdmats_dict["stray"].shape[1]  # number of stray light parameters
    n_source = n_sky + n_inst + n_stray

    fwdmat_final = csc_matrix(([], [], np.zeros(n_source + 1)), shape=(n_data_points, n_source))

    for i in range(n_frames):
        if n_frames == 1:
            fwdmat_final += csc_resize(fwdmats_dict["inst"], n_data_points, n_source, i * n_pixels, 0)
        else:
            fwdmat_final += csc_resize(fwdmats_dict["sky"][i], n_data_points, n_source, i * n_pixels, 0)

    for i in range(n_frames):
        if n_frames > 1:
            fwdmat_final += csc_resize(fwdmats_dict["inst"], n_data_points, n_source, i * n_pixels, n_sky)

    for i in range(n_frames):
        fwdmat_final += csc_resize(fwdmats_dict["stray"].T, n_source, n_data_points, n_sky + n_inst + i * im_size, i * n_pixels).T

    return fwdmat_final


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
    rotmat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return rotmat


def get_sky_source(dimensions: np.ndarray, crota: float = 0, center: np.ndarray = np.array([0, 0]), src_subgrid_fac=2):
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
    src_frame = Trivialframe(["x", "y"])  # saves the coordinate names?
    origin = forward_transform.dot(-0.5 * dimensions) + center

    src_coords = CoordGrid(dimensions, origin, forward_transform, src_frame)
    return SourceGrid(src_coords, None, bin_function_evaluator, nsubgrid=src_subgrid_fac)


def get_detector(
    dimensions: np.ndarray, crota: float = 0, center: np.ndarray = np.array([0, 0]), det_subgrid_fac: int = 3
):
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
    origin = forward_transform.dot(-0.5 * dimensions + center)

    det_coords = CoordGrid(dimensions, origin, forward_transform, det_frame)
    psf_covariance = get_2d_covariance([0.5, 0.5], 0)
    inverse_psf_covariance = np.linalg.inv(psf_covariance)
    return DetectorGrid(
        det_coords,
        [inverse_psf_covariance],
        n_dimensional_gaussian_psf,
        nsubgrid=det_subgrid_fac,
        threshold=1e-3,
        footprint=[25, 25],
    )


def csc_resize(csc, row_size, column_size, row_offset, column_offset):
    """
    Resize the compressed sparse column (csc) matrix into a larger, all-zero csc matrix at a given offset.

    Parameters:
    -----------
    csc: scipy.sparse.csc_matrix
            compressed sparse column matrix of interest to resize
    row_size: int
            Number of rows in the resized output matrix.
            Must satisfy `row_size >= row_offset + csc.shape[0]`.
    column_size: int
            Number of columns in the resized output matrix.
            Must satisfy `column_size >= column_offset + csc.shape[1]`.

    Returns:
    --------
    scipy.sparse.csc_matrix
            A new sparse matrix of shape `(row_size, column_size)` containing `csc` as a submatrix at position
            `(row_offset,column_offset)`, with all other entries equal to zero.

    Notes:
    ------
    `scipy.sparse.csc_matrix((data, indices, indptr), [shape=(M, N)])`
            is the standard CSC representation where the row indices for column i
            are stored in `indices[indptr[i]:indptr[i+1]]` and their corresponding
            values are stored in `data[indptr[i]:indptr[i+1]]`. If the shape parameter
            is not supplied, the matrix dimensions are inferred from the index arrays.
    """
    row_indices = csc.indices + row_offset
    column_indices = csc.indptr

    if column_offset > 0:
        column_indices = np.hstack([np.zeros(column_offset, dtype=np.int64), column_indices])
    n_extra = column_size - column_offset - csc.shape[1]

    if n_extra > 0:
        column_indices = np.hstack([column_indices, csc.indptr[-1] * np.ones(n_extra, dtype=np.int64)])
    print(row_size, column_size, row_offset, column_offset, n_extra, csc.shape, column_indices.shape)

    return csc_matrix((csc.data, row_indices, column_indices), shape=(row_size, column_size))
