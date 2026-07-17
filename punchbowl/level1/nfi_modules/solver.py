import time

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator, lgmres


class NonLinearMapOperator(LinearOperator):
    """
    This operator implements the general linear operator for chi squared
    plus regularization with nonlinear mapping as outlined in Plowman &
    Caspi 2020.
    """

    def setup(self, amat, regmat, map_drvvec, wgtvec, reg_map_drvvec, dtype="float32", reg_fac=1):
        self.amat = amat
        self.regmat = regmat
        self.map_derivative_vec = map_drvvec
        self.weight_vector = wgtvec
        self.reg_map_derivative_vector = reg_map_drvvec
        self.dtype_internal = dtype
        self.reg_fac = reg_fac

    def _matvec(self, vec):
        # Potential GPU acceleration possibility
        chi2term = self.map_derivative_vec * (self.amat.T * (self.weight_vector * (self.amat * (self.map_derivative_vec * vec))))  # A-transpose times A (with non-lin map corrections)
        regterm = self.reg_map_derivative_vector * (self.reg_fac * self.regmat * (self.reg_map_derivative_vector * vec))
        return (chi2term + regterm).astype(self.dtype_internal)

    def _adjoint(self):
        return self


def sparse_nonlinear_map_solver(
    data0,
    errors0,
    amat0,
    guess=None,
    regularization_factor=1,
    forward_func=None,
    derivative_func=None,
    inverse_func=None,
    regularization_matrix=None,
    sparse_matrix_solver = lgmres,
    sqrmap=False,
    reg_func=None,
    deriv_reg_func=None,
    inverse_reg_func=None,
    map_regularization=False,
    adapt_lamda=True,
    solver_tol=1e-3,
    n_iterations=40,
    dtype="float32",
    steps=None,
    flatguess=True,
    chi2_threshold=1,
    chi2_convergence=1e-15,
):
    """
    Subroutine to do the inversion.
    Uses the log mapping and iteration from Plowman & Caspi 2020 to ensure positivity of solutions.

    Parameters:
    -----------
    data0: np.ndarray
        Data values of image(s) for the solver to fit.
    errors0: np.ndarray
        Uncertainties in the data values (`data0`)
    amat0: scipy.sparse.csc_matrix
        Sparse forward matrix that maps coefficients of the solution to the data values.
    guess: np.ndarray, optional
        Initial guess for coefficients.
    reg_fac: float
        Scales the strength of the regularization.
    foward_func: Callable
        The non-linear mapping function. Default is exponential_forward.
    derivative_func: Callable
        The derivative of the forward_func with respect to its argument.
    inverse_func: Callable
        Inverse of the forward_func with respect to its argument.
    regmat: sparse matrix
        Matrix to use for regularization. If passed as None, defaults to diagonal matrix (with variable 
        or constant values depending on other flags). 
    solver: Callable
        Which sparse matrix solver to use. Defaults to `lgmres`
    sqrmap: bool, default=False
        Flag to indicate use of c=s^2 instead of c=e^s.
    reg_func: Callable
        Wrapper function for regularization. Defaults to forward_func (forward operator)
    deriv_reg_func: Callable
        Derivative of regularization function.
    inverse_reg_func: Callable
        Inverse of regularization. 
    map_reg: bool, default=False
        If false, regularizes in the linear non-mapped space.
    adapt_lam: bool, default=True
        Flag to automatically adapt lambda to adjust strength of regularization.
    solver_tol: float, default=1e-3
        Solver tolerance passed to the sparse matrix solver (`solver`). (Equivalent to `atol` of `lgmres`)
    niter: int, default=40
        Number of iterations to to get to non-linear solver.
    dtype: str or numpy.dtype, default="float32"
        Type contraint for computation speed, memory use, and/or calculation precision.
    steps: np.ndarray, optional
        Step sizes to try. Step sizes are fractions of the way to the solution returned by sparse matrix solver. First entry should be zero.
        Default np.array([0.00, 0.05, 0.15, 0.3, 0.5, 0.67, 0.85], dtype=dtype)
    flatguess: bool, default=True
        Indicator of using a "flat guess" i.e. image with constant values, as opposed to using the adjoint (transpose of the forward matrix). 
    chi2_th: float, default=1.0
        Threshold on (reduced) chi-squared, where its considered done.
    conv_chi2: float, default=1e-15
        Convergent chi-squared. The value of the difference between chi-squared iterations to be considered converged.
    Returns:
    --------
    solution: np.ndarray

    chi2: float
        The final chi-sqaured.
    resids: np.ndarray
        Residuals. (Data-Solution)^2/uncertainty^2
    """
    n_data, n_source = amat0.shape

    # Being really careful that everything is the right dtype
    # so (for example) nothing gets promoted to double if dtype is single:
    # TODO: (JK): unit tests will likely help here (or at least some assert statements)
    solver_tol = np.dtype(dtype).type(solver_tol)
    zero = np.dtype(dtype).type(0.0)
    pt5 = np.dtype(dtype).type(0.5)
    two = np.dtype(dtype).type(2.0)
    one = np.dtype(dtype).type(1.0)
    chi2_convergence = np.dtype(dtype).type(chi2_convergence)

    # A collection of example mapping functions.
    def identity_function(s):
        return s  # linear forward

    def inverse_identity_function(s):
        return s  # linear inverse

    def linear_derivative_function(s):
        return one + zero * s  # linear derivative

    def exponential_forward(s):
        return np.exp(s)  # exponential forward

    def exponential_inverse(c):
        return np.log(c)  # exponential inverse

    def exponential_derivative(s):
        return np.exp(s)  # exponential derivative

    def quadratic_forward(s):
        return s * s  # quadratic forward

    def quadratic_inverse(c):
        return c**pt5  # quadratic inverse

    def quadratic_derivative(s):
        return two * s  # quadratic derivative

    # Set-up default functions
    if forward_func is None or derivative_func is None or inverse_func is None:
        if sqrmap:
            forward_func = quadratic_forward
            derivative_func = quadratic_derivative
            inverse_func = quadratic_inverse
        else:
            forward_func = exponential_forward
            derivative_func = exponential_derivative
            inverse_func = exponential_inverse

    if reg_func is None or deriv_reg_func is None or inverse_reg_func is None:
        if map_regularization:
            reg_func = identity_function
            deriv_reg_func = linear_derivative_function
            inverse_reg_func = inverse_identity_function
        else:
            reg_func = forward_func
            deriv_reg_func = derivative_func
            inverse_reg_func = inverse_func

    flat_data = data0.flatten().astype(dtype)
    flat_errs = errors0.flatten().astype(dtype)
    flat_errs[flat_errs == 0] = (0.05 * np.nanmean(flat_errs[flat_errs > 0])).astype(dtype)

    guess0 = amat0.T * (np.clip(flat_data, np.min(flat_errs), None))
    guess0_data = amat0 * guess0
    guess0_norm = np.sum(flat_data * guess0_data / flat_errs**2) / np.sum((guess0_data / flat_errs) ** 2)
    guess0 *= guess0_norm
    guess0 = np.clip(guess0, 0.05 * np.mean(np.abs(guess0)), None).astype(dtype)
    if guess is None:
        guess = guess0
    if flatguess:
        guess = ((1 + np.zeros(n_source)) * np.mean(flat_data) / np.mean(amat0 * (1 + np.zeros(n_source))))
        guess = guess.astype(dtype)
    s_vector = inverse_func(guess).astype(dtype)

    # This is an internal step length limiter to prevent overstepping
    # if the solution starts out in a highly nonlinear region of the mapping
    # function. The solver can still take larger steps since the limiter scales
    # it so that the smallest step size is maxdelta.
    max_delta = inverse_reg_func(np.max(guess0)) - inverse_reg_func(0.25 * np.max(guess0))

    # Try these step sizes at each step of the iteration. Trial Steps are fast compared to computing
    # the matrix inverse, so having a significant number of them is not a problem.
    # Step sizes are specified as a fraction of the full distance to the solution found by the sparse
    # matrix solver (lgmres or bicgstab).
    if steps is None:
        steps = np.array([0.00, 0.05, 0.15, 0.3, 0.5, 0.67, 0.85], dtype=dtype)
    min_step = np.min(steps[1:])
    n_steps = len(steps)
    step_loss = np.zeros(n_steps, dtype=dtype)

    reglam = one
    if regularization_matrix is None and map_regularization:
        regularization_matrix = diags(one / inverse_reg_func(guess0) ** two)
    if adapt_lamda and map_regularization:
        reglam = (np.dot((regularization_matrix * s_vector), derivative_func(s_vector) * (amat0.T * (1 / flat_errs)))
                  / np.dot((regularization_matrix * s_vector), (regularization_matrix * s_vector)))
    if regularization_matrix is None and not map_regularization:
        regularization_matrix = diags(1 / guess0 ** 2)
    if adapt_lamda and not map_regularization:
        reglam = np.dot(
            derivative_func(s_vector) * (regularization_matrix * guess), derivative_func(s_vector) * (amat0.T * (1 / flat_errs))
        ) / np.dot(derivative_func(s_vector) * (regularization_matrix * guess), derivative_func(s_vector) * (regularization_matrix * guess))

    
    regularization_matrix = regularization_factor * regularization_matrix * reglam
    weights = (1 / flat_errs**2).astype(dtype)  # The weights are the errors...

    nlmo = NonLinearMapOperator(dtype=dtype, shape=(n_source, n_source))
    nlmo.setup(amat0, regularization_matrix, derivative_func(s_vector), weights, deriv_reg_func(s_vector), reg_fac=regularization_factor)

    # --------------------- Now do the iteration:
    setup_timer = 0
    solver_timer = 0
    stepper_timer = 0
    for i in range(n_iterations):
        tsetup = time.time()
        # Setup intermediate matrices for solution:
        d_guess = derivative_func(s_vector)     # derivative guess
        d_reg_guess = deriv_reg_func(s_vector)  # derivative of regularization guess
        bvec = d_guess * amat0.T.dot(
            weights * (flat_data - amat0 * (forward_func(s_vector) - s_vector * derivative_func(s_vector)))
        )
        if not map_regularization:
            bvec -= d_reg_guess * (regularization_factor * regularization_matrix * (reg_func(s_vector) - s_vector * deriv_reg_func(s_vector)))
        setup_timer += time.time() - tsetup

        tsolver = time.time()
        # Run sparse matrix solver:
        nlmo.map_derivative_vec, nlmo.reg_map_derivative_vector = d_guess, d_reg_guess
        svec2 = sparse_matrix_solver(
            nlmo, bvec.astype(dtype), s_vector.astype(dtype), store_outer_Av=False, atol=solver_tol.astype(dtype)
        )
        svec2 = svec2[0]
        solver_timer += time.time() - tsolver

        tstepper = time.time()
        deltas = svec2 - s_vector
        if np.max(np.abs(deltas)) == 0:
            break  # This also means we've converged.

        # Rescale the deltas so they don't exceed maxdelta at the smallest step size:
        deltas *= np.clip(np.clip(np.max(np.abs(deltas)), None, max_delta / min_step) / np.max(np.abs(deltas)), 0, 1)

        # Try the step sizes:
        for j in range(n_steps):
            stepguess = forward_func(s_vector + steps[j] * deltas)
            stepguess_reg = reg_func(s_vector + steps[j] * deltas)
            stepresid = (flat_data - amat0 * stepguess) * weights ** pt5
            step_loss[j] = (
                np.dot(stepresid, stepresid) / n_data
                + np.sum(stepguess_reg.T * (regularization_factor * regularization_matrix * stepguess_reg)) / n_data
            )

        best_step = np.nanargmin(step_loss[1:n_steps]) + 1  # First step is zero for comparison purposes...
        chi20 = np.sum(weights * (flat_data - amat0 * (forward_func(s_vector))) ** two) / n_data
        reg0 = np.sum(reg_func(s_vector.T) * (regularization_factor * regularization_matrix * (reg_func(s_vector)))) / n_data

        # Update the solution with the step size that has the best Chi squared:
        s_vector = s_vector + steps[best_step] * deltas
        reg1 = np.sum(reg_func(s_vector.T) * (regularization_factor * regularization_matrix * (reg_func(s_vector)))) / n_data
        resids = weights * (flat_data - amat0 * (forward_func(s_vector))) ** two
        chi21 = np.sum(weights * (flat_data - amat0 * (forward_func(s_vector))) ** two) / n_data
        stepper_timer += time.time() - tstepper

        if np.abs(step_loss[0] - step_loss[best_step]) < chi2_convergence or chi21 < chi2_threshold:
            break  # Finish the iteration if chi squared isn't changing

    return forward_func(s_vector), chi21, resids
