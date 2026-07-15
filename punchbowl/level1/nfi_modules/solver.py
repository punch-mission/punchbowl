from scipy.sparse.linalg import LinearOperator
import time
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import lgmres

class NlmapOperator(LinearOperator):
	"""
	This operator implements the general linear operator for chi squared
	plus regularization with nonlinear mapping as outlined in Plowman &
	Caspi 2020.
	"""

	def setup(self,amat,regmat,map_drvvec,wgtvec,reg_map_drvvec,dtype="float32",reg_fac=1):
		self.amat = amat
		self.regmat = regmat
		self.map_drvvec = map_drvvec
		self.wgtvec = wgtvec
		self.reg_map_drvvec = reg_map_drvvec
		self.dtype_internal=dtype
		self.reg_fac = reg_fac

	def _matvec(self,vec):
		chi2term = self.map_drvvec*(self.amat.T*(self.wgtvec*(self.amat*(self.map_drvvec*vec))))
		regterm = self.reg_map_drvvec*(self.reg_fac*self.regmat*(self.reg_map_drvvec*vec))
		return (chi2term+regterm).astype(self.dtype_internal)

	def _adjoint(self):
		return self

def sparse_nlmap_solver(data0, errors0, amat0, guess=None, reg_fac=1, forward_func=None, derivative_func=None, inverse_func=None, regmat=None, silent=False,
						solver=None, sqrmap=False, reg_func=None, deriv_reg_func=None, inverse_reg_func=None, map_reg=False, adapt_lam=True,
						solver_tol = 1.0e-3, niter=40, dtype="float32", steps=None, precompute_ata=False, flatguess=True, chi2_th=1.0,
						store_outer_Av=False, conv_chi2 = 1.0e-15):
	"""
	Subroutine to do the inversion. 
	Uses the log mapping and iteration from Plowman & Caspi 2020 to ensure positivity of solutions.

	Parameters:
	-----------
	data0: np.ndarray

	errors0: np.ndarray

	amat0: scipy.sparse.csc_matrix

	guess: np.ndarray

	reg_fac: Callable

	foward_func: Callable

	derivative_func: Callable

	inverse_func: Callable

	regmat: sparse matrix

	silent: bool, optional, default=False

	solver: Callable

	sqrmap: bool, default=False

	reg_func: Callable

	deriv_reg_func: Callable

	inverse_reg_func: Callable

	map_reg: bool, default=False

	adapt_lam: bool, default=True

	solver_tol: float, default=1e-3

	niter: int, default=40

	dtype: str or numpy.dtype, default="float32"

	steps: np.ndarray, optional

	precompute_ata: bool, default=False

	flatguess: bool, default=True

	chi2_th: float, default=1.0

	store_outer_Av: bool, default=False

	conv_chi2: float, default=1e-15

	Returns:
	--------
	solution: np.ndarray

	chi2: float

	resids: np.ndarray
	
	"""
	[n_data, n_source] = amat0.shape

	# Being really careful that everything is the right dtype
	# so (for example) nothing gets promoted to double if dtype is single:
	# TODO: (JK): unit tests will likely help here (or at least some assert statements)
	solver_tol = np.dtype(dtype).type(solver_tol)
	zero = np.dtype(dtype).type(0.0)
	pt5 = np.dtype(dtype).type(0.5)
	two = np.dtype(dtype).type(2.0)
	one = np.dtype(dtype).type(1.0)
	conv_chi2 = np.dtype(dtype).type(conv_chi2)

	# A collection of example mapping functions.
	def identity_function(s): return s # linear forward
	def inverse_identity_function(s): return s # linear inverse
	def linear_derivative_function(s): return one + zero*s # linear derivative
	def exponential_forward(s): return np.exp(s) # exponential forward
	def exponential_inverse(c): return np.log(c) # exponential inverse
	def exponential_derivative(s): return np.exp(s) # exponential derivative
	def quadratic_forward(s): return s*s # quadratic forward
	def quadratic_inverse(c): return c**pt5 # quadratic inverse
	def quadratic_derivative(s): return two*s # quadratic derivative

	if forward_func is None or derivative_func is None or inverse_func is None:
		if sqrmap:
			[forward_func,derivative_func,inverse_func] = [quadratic_forward,quadratic_derivative,quadratic_inverse]
		else:
			[forward_func,derivative_func,inverse_func] = [exponential_forward,exponential_derivative,exponential_inverse]

	if reg_func is None or deriv_reg_func is None or inverse_reg_func is None:
		if map_reg:
			[reg_func,deriv_reg_func,inverse_reg_func] = [identity_function,linear_derivative_function,inverse_identity_function]
		else:
			[reg_func,deriv_reg_func,inverse_reg_func] = [forward_func,derivative_func,inverse_func]

	if solver is None:
		solver = lgmres

	flat_data = data0.flatten().astype(dtype)
	flat_errs = errors0.flatten().astype(dtype)
	flat_errs[flat_errs == 0] = (0.05*np.nanmean(flat_errs[flat_errs > 0])).astype(dtype)

	guess0 = amat0.T*(np.clip(flat_data,np.min(flat_errs),None))
	guess0_data = amat0*(guess0)
	guess0_norm = np.sum(flat_data*guess0_data/flat_errs**2)/np.sum((guess0_data/flat_errs)**2)
	guess0 *= guess0_norm
	guess0 = np.clip(guess0,0.05*np.mean(np.abs(guess0)),None).astype(dtype)
	if guess is None:
		guess = guess0
	if flatguess:
		guess = ((1+np.zeros(n_source))*np.mean(flat_data)/np.mean(amat0*(1+np.zeros(n_source)))).astype(dtype)
	s_vector = inverse_func(guess).astype(dtype)

	# This is an internal step length limiter to prevent overstepping
	# if the solution starts out in a highly nonlinear region of the mapping
	# function. The solver can still take larger steps since the limiter scales
	# it so that the smallest step size is maxdelta.
	max_delta = inverse_reg_func(np.max(guess0))-inverse_reg_func(0.25*np.max(guess0))

	# Try these step sizes at each step of the iteration. Trial Steps are fast compared to computing
	# the matrix inverse, so having a significant number of them is not a problem.
	# Step sizes are specified as a fraction of the full distance to the solution found by the sparse
	# matrix solver (lgmres or bicgstab).
	if steps is None:
		steps = np.array([0.00, 0.05, 0.15, 0.3, 0.5, 0.67, 0.85],dtype=dtype)
	min_step = np.min(steps[1:])
	n_steps = len(steps)
	step_loss = np.zeros(n_steps,dtype=dtype)

	reglam = one
	if regmat is None and map_reg:
		regmat = diags(one/inverse_reg_func(guess0)**two)
	if adapt_lam and map_reg:
		reglam = (np.dot((regmat*s_vector),
							derivative_func(s_vector)*(amat0.T*(1.0/flat_errs)))/
							np.dot((regmat*s_vector),(regmat*s_vector)))
	if regmat is None and not map_reg:
		regmat = diags(1.0/(guess0)**2)
	if adapt_lam and not map_reg:
		reglam = (np.dot(derivative_func(s_vector)*(regmat*guess),
							derivative_func(s_vector)*(amat0.T*(1.0/flat_errs)))/
							np.dot(derivative_func(s_vector)*(regmat*guess),derivative_func(s_vector)*(regmat*guess)))

	# Still appears to be some issue with this regularization factor?
	regmat = reg_fac*regmat*reglam
	weights = (1.0/flat_errs**2).astype(dtype) # The weights are the errors...

	if(not(precompute_ata)):
		nlmo = NlmapOperator(dtype=dtype, shape=(n_source, n_source))
		nlmo.setup(amat0,regmat,derivative_func(s_vector),weights,deriv_reg_func(s_vector),reg_fac=reg_fac)

	# --------------------- Now do the iteration:
	tstart = time.time()
	setup_timer = 0
	solver_timer = 0
	stepper_timer = 0
	for i in range(niter):
		tsetup = time.time()
		# Setup intermediate matrices for solution:
		dguess = derivative_func(s_vector)
		dregguess = deriv_reg_func(s_vector)
		bvec = dguess*amat0.T.dot(weights*(flat_data-amat0*(forward_func(s_vector)-s_vector*derivative_func(s_vector))))
		if(map_reg==False): bvec -= dregguess*(reg_fac*regmat*(reg_func(s_vector)-s_vector*deriv_reg_func(s_vector)))
		setup_timer += time.time()-tsetup

		tsolver = time.time()
		# Run sparse matrix solver:
		[nlmo.map_drvvec,nlmo.reg_map_drvvec] = [dguess,dregguess]
		svec2 = solver(nlmo,bvec.astype(dtype),s_vector.astype(dtype),store_outer_Av=False,atol=solver_tol.astype(dtype))
		svec2 = svec2[0]
		solver_timer += time.time()-tsolver

		tstepper = time.time()
		deltas = svec2-s_vector
		if np.max(np.abs(deltas)) == 0:
			break # This also means we've converged.

		# Rescale the deltas so they don't exceed maxdelta at the smallest step size:
		deltas *= np.clip(np.clip(np.max(np.abs(deltas)),None,max_delta/min_step)/np.max(np.abs(deltas)),0,1)

		# Try the step sizes:
		for j in range(n_steps):
			stepguess = forward_func(s_vector+steps[j]*(deltas))
			stepguess_reg = reg_func(s_vector+steps[j]*(deltas))
			stepresid = (flat_data-amat0*(stepguess))*weights**pt5
			step_loss[j] = np.dot(stepresid,stepresid)/n_data + np.sum(stepguess_reg.T*(reg_fac*regmat*(stepguess_reg)))/n_data

		best_step = np.nanargmin((step_loss)[1:n_steps])+1 # First step is zero for comparison purposes...
		chi20 = np.sum(weights*(flat_data-amat0*(forward_func(s_vector)))**two)/n_data
		reg0 = np.sum(reg_func(s_vector.T)*(reg_fac*regmat*(reg_func(s_vector))))/n_data

		# Update the solution with the step size that has the best Chi squared:
		s_vector = s_vector+steps[best_step]*(deltas)
		reg1 = np.sum(reg_func(s_vector.T)*(reg_fac*regmat*(reg_func(s_vector))))/n_data
		resids = weights*(flat_data-amat0*(forward_func(s_vector)))**two
		chi21 = np.sum(weights*(flat_data-amat0*(forward_func(s_vector)))**two)/n_data
		stepper_timer += time.time()-tstepper

		if(np.abs(step_loss[0]-step_loss[best_step]) < conv_chi2 or chi21 < chi2_th):
			break # Finish the iteration if chi squared isn't changing

	return forward_func(s_vector), chi21, resids
