import numpy as np
from scipy.sparse import csc_matrix, diags

from punchbowl.level1.nfi_modules.fwdmats import assemble_nfi_fwdmats
from punchbowl.level1.nfi_modules.solver import sparse_nonlinear_map_solver


def reconstruct_nfi_straylight(
    data,
    errs,
    amats,
    good_dat,
    bin_fac=4,
    errfac_systematic=0.01,
    mask_source=False,
    solver_tol=2.5e-5,
    sky_reg=1,
    inst_reg=1,
    stray_reg=0.001,
    datanorm=1e-10,
):
    """
    This is the routine that does the actual inversion of the sky and stray light components

    Parameters:
    -----------
    data: np.array
            The images to invert, dimensions n_img, nx, ny
    errs: np.ndarray
            Uncertainties corresponding to the images
    amats: dict
            dictionary containing the forward matrices for the sky, (per-pixel) instrument, and stray light
            sources.
            Created by `fwdmats.generate_nfi_forward_matrices`
    good_dat: np.ndarray
            Array flagging which data are good to use in the inversion, same shape as data
    bin_fac: int, optional
            how much to bin down the data for speed (default: 4). fwdmats.generate_nfi_fwdmats
                            must be called with the same bin_fac.
    errfac_systematic: float, optional, default 0.0.1
            An additional uncertainty of this factor multiplied by the data is added to the errors.
    solver_tol: float, optional
            Tolerance for the solver, default 2.5e-5
    sky_reg: float, optional
            Regularization factor for the sky source, larger values are a heavier penalty; default 1
    inst_reg: float, optional
            Regularization factor for per-pixel instrument source, default 1
    stray_reg: float, optional
            Regularization factor for the disk stray light functions, default 0.001
    mask_source: bool, optional
            Attempts to mask off source coefficients with no connection to valid data. Not working.

    Returns:
    --------
    soln_sky: np.ndarray
            Sky model (background stars and f-corona) solution.
    soln_ins: np.ndarray
            Instrument based stray light solution.
    soln_stray: list of np.ndarray
            Reconstructed stray light model for each frame.
    data_binned: np.ndarray
            The input data, after being normalized. -- i.e. numerically equivalent to the original passed `data`,
            but returned for post-processing/sanity-check comparison.
    """
    fwdmat = assemble_nfi_fwdmats(amats)

    n_frames, nx, ny = data.shape[0], round(data.shape[1]), round(data.shape[2])
    im_size = nx
    n_pixels = nx * ny
    dims = np.array([nx, ny], dtype=np.int32)
    n_stray_coeffs = amats["stray"].shape[1]
    n_sky = n_pixels
    n_instr = n_pixels * (n_frames > 1)
    n_stray = n_frames * n_stray_coeffs
    n_source = n_sky + n_instr + n_stray

    data_bin = [d / datanorm for d in data]
    err_bin = [e / datanorm for e in errs]
    mask_bin = [g for g in good_dat]
    good_data_mask = np.hstack([m.flatten() for m in mask_bin])

    # Mask matrices
    source_mask_matrix = mask_sources(fwdmat)
    data_mask_matrix = mask_data(fwdmat, good_data_mask)

    # Regularization vectors/matrix
    reg_vector = np.ones(fwdmat.shape[1])
    reg_vector[0:n_sky] = sky_reg
    if n_instr > 0:
        reg_vector[n_sky : n_sky + n_instr] *= inst_reg
    reg_vector[n_sky + n_instr :] *= stray_reg
    reg_matrix = diags(reg_vector)

    # Apply masks to relevant data
    fwdmat_masked = data_mask_matrix * fwdmat * source_mask_matrix.T
    regmat_masked = source_mask_matrix * reg_matrix * source_mask_matrix.T

    flat_data = np.hstack([d.flatten() for d in data_bin])
    flat_err = np.hstack([e.flatten() for e in err_bin]) + errfac_systematic * np.abs(flat_data)
    solution = sparse_nonlinear_map_solver(
        data_mask_matrix * flat_data,
        data_mask_matrix * flat_err,
        fwdmat_masked,
        adapt_lamda=False,
        regularization_factor=1,
        dtype="float32",
        n_iterations=40,
        solver_tol=solver_tol,
        sqrmap=False,
        flatguess=True,
        regularization_matrix=regmat_masked,
    )  # , guess = np.ones(reg_guess.size))

    soln = source_mask_matrix.T * solution[0]
    if n_instr > 0:
        soln_sky = datanorm * (amats["sky"][0] * (soln[0:n_pixels])).reshape(dims)
        soln_ins = datanorm * (amats["inst"] * (soln[n_pixels : 2 * n_pixels])).reshape(dims)
    else:
        soln_sky = datanorm * (amats["inst"] * (soln[0:n_pixels])).reshape(dims)
        soln_ins = np.zeros(dims)
    soln_stray = []
    for i in range(n_frames):
        soln_stray.append(
            datanorm
            * (
                amats["stray"]
                * (soln[n_sky + n_instr + i * n_stray_coeffs : n_sky + n_instr + (i + 1) * n_stray_coeffs])
            ).reshape(dims)
        )

    return soln_sky, soln_ins, soln_stray, np.array(data_bin) * datanorm


def mask_sources(amat_in, mask_lvl=None):
    """
    Mask sources that aren't present in the data.

    Parameters:
    -----------
    amat_in: scipy.sparse.csc_matrix
            Forward matrix of interest with sources to filtered.
            Created by `fwdmats.assemble_nfi_fwdmats`

    mask_lvl: float, optional, default = None
            Minimum column-sum threshold for a source to be kept.
            Sources with column sum >= `mask_lvl` are retained.
            If `None` (default), the threshold is set adaptively to 5% of the mean column
            sum of `amat_in`.

    Returns:
    --------
    mask_matrix_sparse: scipy.sparse.cscmatrix
            (Sparse) matrix for source mask.
            Matrix multiply with compatible matrix (e.g. `amat_in`) to filter out the masked-out
            sources.
    """
    column_sums = np.sum(amat_in, axis=0).A1
    if mask_lvl is None:
        mask_lvl = 0.05 * np.mean(column_sums)

    source_mask = column_sums >= mask_lvl
    n_source_in = len(source_mask)
    n_source_out = np.sum(source_mask)
    mask_indices = np.arange(n_source_in, dtype=np.uint64)

    input_mask_indices = mask_indices[source_mask]
    output_mask_indices = np.arange(n_source_out, dtype=np.uint64)

    mask_matrix_vals = np.ones(n_source_out, dtype=np.float32)
    mask_matrix_sparse = csc_matrix(
        (mask_matrix_vals, (output_mask_indices, input_mask_indices)), shape=[n_source_out, n_source_in]
    )
    return mask_matrix_sparse


def mask_data(amat_in, good_data_mask, mask_lvl=None):
    """
    Mask data that aren't connected to the sources (or to anything).

    Parameters:
    -----------
    amat_in: scipy.sparse.csc_matrix
            Forward matrix of interest with sources to filtered.
            Created by `fwdmats.assemble_nfi_fwdmats`
    good_data_mask: np.array
            Per-row boolean mask indicating which rows are considered
            valid a priori. Rows marked False are always excluded,
            regardless of `mask_lvl`.
    mask_lvl: float, optional, default=None
            Minimum row-sum threshold for a source to be kept.
            Data with row sum >= `mask_lvl` are retained.
            If `None` (default), the threshold is set adaptively to 5% of the mean row
            sum of `amat_in`.

    Returns:
    --------
    mask_matrix_sparse: scipy.sparse.csc_matrix
            (Sparse) matrix for masking the invalid data.
    """
    row_sums = np.sum(amat_in, axis=1).A1
    if mask_lvl is None:
        mask_lvl = 0.05 * np.mean(row_sums)

    data_mask = good_data_mask * (row_sums >= mask_lvl)
    n_data_in = len(data_mask)
    n_data_out = np.sum(data_mask)

    mask_indices = np.arange(n_data_in, dtype=np.uint64)
    input_mask_indices = mask_indices[data_mask]
    output_mask_indices = np.arange(n_data_out, dtype=np.uint64)

    mask_matrix_vals = np.ones(n_data_out, dtype=np.float32)
    mask_matrix_sparse = csc_matrix(
        (mask_matrix_vals, (output_mask_indices, input_mask_indices)), shape=[n_data_out, n_data_in]
    )
    return mask_matrix_sparse
