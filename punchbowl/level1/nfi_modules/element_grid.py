import numpy as np

from punchbowl.level1.nfi_modules.util import forward_rolling_transpose, roll_transpose_from_numpy_indices

TINY = 1.0e-4


class ElementGrid:
    """
    This ElementGrid class builds on CoordGrid to specify a grid of basis or detector
    elements (e.g., the spatial response functions of an imaging detector, including
    their pixel boxes and PSF). It is the exemplar class for the source and detector
    objects used by get_sparse_response_matrix.

    Elements are addressed by a single index apiece, but
    this is trivial to multiplex under the hood using, for instance,
    i = i0*nj*nk+j0*nk+k0. Numpy's ravel_index, unravel_multi_index, and
    flatten functions provide exactly this functionality. The intention of this class
    is to implement the functionality needed by get_sparse_response_matrix for typical
    detector arrays e.g., 2D imagers or 3D spectrographs, as well as for gridded arrays
    of source elements. It should be fairly powerful and extensible.

    Attributes
    ----------
    threshold:
        Final threshold for keeping points when evaluating the response/basis functions.
    coords:
        Information about the coordinate system of the output of elements and
        the input to response. Implemented as an instance of coordgrid.
    params:
        Parameters or anything else used to evaluate the response function.
    function_evaluator:
        The response function evaluator.
        The callable should take indices, coordinates, and `params` as input, and return the response
        for the point(s) of interest
    n_elements: int
        The number of elements in the element_grid, as well as the number
        of unique element id/indices.
    n_subgrid:
        To take into account subgrid-scale effects, evaluate the response/basis functions.
        at this multiple of the grid scale.
    subgrid:
        Subgrid made by `CoordGrid.subgrid` factored `n_subgrid`
    eval_subgrid:
        Grid returned from `get_eval_grid()` method
    stencil:
        In addition to the footprint, a stencil is computed to
        determine which grid points to use evaluate around the input point.
    n_addresses: int
        The number of element addresses in the element grid. These are
        how the elements are accessed via the elements method. In
        basic use these will be the same as the element indices,
        but there are use cases where other addressing modes are
        desired. The addresses are generally expected to be positive integers.

    Methods
    -------
    get_eval_grid():
        Returns a grid for evaluating the response/basis functions
    get_n_addresses():
        Returns the number of addresses.
    evaluate_basis_at_point(point):
        Evaluate the source/basis function at a given point.
    elements(point):
        Returns the elements addressed by a given index
    response(point):
        Returns the response of the elements to a delta function source at a given
        point

    Notes:
    ------
    There is an issue with rounding and floating point jitter for aligned grids that
    share some ratios in their spacing. We add this small offset when discretizing
    grid indices to avoid this, which is a bit of a hack..

    On a related note, there are some intrinsic issues with even nsubgrid, which
    can produce subpixel shifts in certain situations. To see why this is the case
    consider convolution by a boxcar kernel with an even width: Such kernels are
    intrinsically asymmetric... Recommend setting det_subgrid_fac to an odd
    number over 1.
    """

    def __init__(
        self, grid, params, function_evaluator, footprint=None, stencil_threshold=5e-4, nsubgrid=3, threshold=0.005
    ):
        """
        Set up the elements.

        Parameters
        -----------
        grid:
            A CoordGrid
        params:
            Parameters or anything else used to evaluate the response function.
        function_evaluator : Callable
            The response function evaluator.
            The callable should take indices, coordinates, and `params` as input, and return the response
            for the point(s) of interest

        footprint: int, optional
            How far away from the input point to evaluate the response functions,
            in grid points.
            (Recommended: at least 21 points, if not None)
        stencil_threshold: float, default: .0005
            In addition to the footprint, a stencil is computed to
            determine which grid points to use evaluate around the input point.

            An initial check of the response function (with evaluation point
            at the origin) is made, and points falling below this threshold are
            omitted. Would probably be better to check the stencil for every
            element's location, although that would be slower...
        nsubgrid: int,  default: 3
            To take into account subgrid-scale effects, evaluate the response/basis functions.
            at this multiple of the grid scale.
        threshold: float, default: 0.005
            Final threshold for keeping points when evaluating the response/basis functions.

        """
        self.threshold = threshold
        self.coords = grid
        self.params = params
        self.function_evaluator = function_evaluator

        self.n_elements = np.prod(grid.dims)
        self.n_subgrid = np.broadcast_to(nsubgrid, grid.dims.shape)

        self.subgrid = grid.subgrid(factor=self.n_subgrid)
        self.eval_grid = self.get_eval_grid()

        # Generate the stencil:
        if footprint is None:
            footprint_offset = np.ceil(10 * self.n_subgrid / 2).astype(np.int32)
        else:
            footprint_offset = np.ceil((footprint / self.n_subgrid - self.n_subgrid) / 2).astype(np.int32)

        self.stencil = (
            roll_transpose_from_numpy_indices(self.n_subgrid + 2 * footprint_offset)
            - footprint_offset
            - 0.5 * (nsubgrid - 1)
        )
        vals = self.evaluate_basis_at_point(self.coords.origin)[0].flatten()
        self.stencil = np.vstack(
            [x.flatten()[vals >= stencil_threshold] for x in list(forward_rolling_transpose(self.stencil))]
        ).T

        # Set the number of addresses:
        self.n_addresses = self.get_n_addresses()

    def get_eval_grid(self):
        """
        Returns a grid for evaluating the response/basis functions
        Standard evaluation grid is the same as the subgrid.
        Note: indices for the eval_grid are assumed to be the same as the subgrid.

        Returns
        -------
        CoordGrid object
            Evaluation grid (which is assumed to be the same as the subgrid)
        """
        return self.coords.subgrid(factor=self.n_subgrid)

    def get_n_addresses(self):
        """
        Returns the number of addresses.
        Standard assumption is number of addresses is same as number of elements.

        Returns
        -------
        int
            The number of addresses (which is assumed to be the same as the number of elements)
        """
        return self.n_elements

    def evaluate_basis_at_point(self, point):
        """
        Evaluate the source/basis function at a given point.

        Parameters
        ----------
        point : np.array
            Point(s) of interest

        Returns
        -------
        vals : np.array
            Response of evaluation point(s)
        output_coords : np.array
            Coordinates of the evaluation point(s)
        """

        # Find where the point is relative to the subgrid
        subpt = self.subgrid.get_indices_from_coordinates(point)

        # Find the stencil evaluation indices (which are registered to the subgrid)
        # in the vicinity of this point.
        subinds = np.round(self.stencil + subpt + TINY)

        # Get the coordinates of these evaluation indices in the evaluation coordinate frame
        # and the subgrid coordinates:
        [eval_coords, output_coords] = [
            self.eval_grid.get_coordinates_from_indices(subinds),
            self.subgrid.get_coordinates_from_indices(subinds),
        ]

        # Compute the response of these evaluation points to the input point, and their
        # coordinates:
        return self.function_evaluator(
            self.eval_grid.get_coordinates_from_indices(subpt), eval_coords, self.params
        ), output_coords

    def get_element_properties_at_point(self, index):
        """
        Returns the properties of the element(s) at `index`.

        Parameters
        ----------
        index : np.ndarray
            Index of interest.

        Returns
        --------
        pnts : np.ndarray
            The coordinate points at which the these elements are non-zero,
            in the element_grid's coordinate frame. Dimensions should be
            npts by ndim, where ndim is the dimensionality of the
            source coordinate system.
        vals : np.ndarray
            The values of the elements' 'basis' function(s) at
            those coordinates. Dimensions are npts.
        coords: np.ndarray
            Indices of the element(s) corresponding to each of those
            point/value pairs. Most often these will all be the same as i,
            but they don't have to be. Dimensions are npts.
        """
        # Map the address to a coordinate and run the element_grid's evaluator:
        [vals, coords] = self.evaluate_basis_at_point(
            self.coords.get_coordinates_from_indices(np.array(np.unravel_index(index, self.coords.dims)))
        )
        # Return the result, element index is same as input address:
        return index + 0 * vals.astype(np.int32), vals, coords

    # Run the evaluator for the given point and compute the output value and indices to flatinds:
    def response(self, point):
        """
        Returns the response of the elements to a delta function source at a given
        point in the ElementGrid's coordinate frame.

        Parameters
        ----------
        point : np.ndarray
            point of interest

        Returns
        -------
        elms : np.ndarray
            The indices of every element in the grid which has a non-zero
            response to a delta function source at the given point.
        vals : np.ndarray
            The values of each of those responses. Same dimensions as `elms`.
        """
        return self.coords.get_flattened_indices(*self.evaluate_basis_at_point(point), threshold=self.threshold)


class DetectorGrid(ElementGrid):
    """The detector grid is a straight implementation of the base class: ElementGrid"""


class SourceGrid(ElementGrid):
    """
    The source grid is the same as the base class except that the evaluation grid for
    the source basis functions is a subgrid consisting of indices rather than using a
    physically dimensioned coordinate system.

    We use a fun trick with CoordGrid's identify and subgrid methods to create this.

    Method
    ------
    get_eval_grid():
        Returns subgrid with indices instead of dimensioned coordinate system.
    """

    def get_eval_grid(self):
        return self.coords.identity().subgrid(factor=self.n_subgrid)
