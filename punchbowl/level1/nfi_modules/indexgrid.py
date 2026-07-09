"""
There is an issue with rounding and floating point jitter for aligned grids that
share some ratios in their spacing. We add this small offset when discretizing
grid indices to avoid this, which is a bit of a hack...
"""
TINY = 1.0e-4

import numpy as np
from nfi_modules.util import multivector_matrix_multiply

class CoordGrid:
    """
    This implements the notion of a grid of coordinate points that are meant to be indexed by integers.
    The basic class includes implementation for an affine (linear transformation) grid, but more general
    coordinate systems will not be hard to extend. The transformation from the indices into the coordinate
    points here should be reversible or the inds and flatinds methods will not work.

    Attributes
    ----------
    dims : np.array
        The dimensions of the source grid, a 1-D array containing [nx0,nx1,...]
    origin : np.array
        The location of the center of the [0,0,...] pixel in the grid
    fwd : np.array
        The forward transformation matrix
    inv : np.array
        The inverse of the forward transformation matrix
    frame : TrivialFrame
        Information about the coordinate frame. Not used by the CoordGrid itself, but 
            may be used by other routines that want to transform between grids.

    Methods
    -------
    get_grid_inverse()
        Routine to set up the parameters of the grid inverse (from coordinates to indices)
        operation.
    subgrid(fac=2)
        Get a grid that's clocked to this grid, but is higher resolution by an integer factor.
    identity()
        Get the identity grid for this grid.
    get_indices_from_coordinates(coords)
        Returns indices given a set of coordinates.
    get_coordinates_from_indices(inds)
        Returns coordinates given a set of indices.
    get_flattened_indices(vals,coords,thold=0)
        Returns the 'flattened' indices given a set of coordinates.
    """

    def __init__(self, dims, origin, forward, frame, inverse=None):
        """
        Set up the grid. 

        Parameters
        ----------
        dims : np.array
            The dimensions of the source grid, a 1-D array containing [nx0,nx1,...]
        origin : np.array 
            The location of the center of the [0,0,...] pixel in the grid
        forward :  np.array
            The information needed to define the forward transformation (i.e., from 
            indices to coordinates), in addition to the origin. For the affine 
            transformation, this is an nd by nd matrix, where nd is the number of dimensions.
        frame : 
            Information about the coordinate frame. Not used by the CoordGrid itself, but 
            may be used by other routines that want to transform between grids.
        inverse : optional 
            The information needed to define the inverse transformation (i.e., from 
            coordinates to indices). 
            Optional -- For the base (affine) implementation, this is computed from fwd using np.linalg.inv.
        """
        self.dims = dims
        self.origin = origin
        self.fwd = forward
        self.inv = inverse
        self.frame = frame
        if self.inv is None:
            self.inv = self.get_grid_inverse()


    def get_grid_inverse(self):
        """
        Routine to set up the parameters of the grid inverse (from coordinates to indices)
        operation. This default assumes an affine transformation and uses the matrix inverse.

        Returns:
        --------
        inverse_fwd : np.array
            Inverse of forward transform (`self.fwd`)
        """
        return np.linalg.inv(self.fwd)

    def subgrid(self,factor=2):
        """
        Get a grid that's clocked to this grid, but is higher resolution by an integer factor.
        
        Parameters
        ----------
        factor: int, default = 2
            Factor to scale CoordGrid to higher resolution.

        Returns
        -------
        CoordGrid
            Grid with new resolution, scaled by passed factor value
        """
        return CoordGrid(self.dims * factor, self.get_coordinates_from_indices(0.5 / factor - 0.5 + 0.0 * self.dims), self.fwd / factor, self.frame)


    def identity(self):
        """
        Get the identity grid for this grid -- i.e., it maps indices to themselves.
        Indices are coordinates too, ya know! Pairs well with subgrid.

        Returns
        -------
        CoordGrid
            Identity grid (mapped indices to themselves) of grid object 
        """
        return CoordGrid(self.dims, 0.0 * self.dims, np.diag(1 + 0.0 * self.dims), np.arange(len(self.dims)))


    def get_indices_from_coordinates(self,coords):
        """
        Returns indices given a set of coordinates. Does no discretize for various reasons.
        Order is reversed, and the inv operator transposed, due to how numpy array broadcasting
        and matrix operations interact. Because coords returns coordinates reference to the
        centers of the grid elements, discretization of these indices should be done with
        rounding, not flooring (see flatinds). The domain of each grid element extends
        0.5 grid spacings to each side.

        Parameter
        ---------
        coords : 
            Coordinates of interest
        Returns
        -------
        np.array
            Indices of given coordinates.
        """
        return multivector_matrix_multiply(self.inv,coords-self.origin)


    def get_coordinates_from_indices(self,inds):
        """
        Returns coordinates given a set of indices. Coordinates returned for an integer index
        are for the center of the grid element, not its corner.

        Parameters
        ---------
        inds : np.array
            Indices of interest

        Returns
        -------
        np.array
            The coordinates of the associated indices (with respect to the origin).
        """
        return multivector_matrix_multiply(self.fwd, inds)+self.origin


    def get_flattened_indices(self,vals,coords,threshold=0):
        """
        Returns the 'flattened' indices given a set of coordinates. Does discretize (because it has to).
        Also discards out-of-bounds points. Because of this, there's an accompanying vals array that can
        be used to account for the discarding.

        Parameters
        ----------
        vals : 

        coords : 

        threshold : int, default = 0

        Returns
        -------
        elms : np.ndarray
            The indices of every element in the grid which has a non-zero
            response to a delta function source at the given point.
        vals : np.ndarray
            The values of each of those responses. Same dimensions as `elms`.
        """
        inds = list(np.round(self.get_indices_from_coordinates(coords)+TINY).T.astype(np.int32))
        keeps = vals>threshold
        for j in range(len(self.dims)): 
            keeps *= (inds[j] >= 0)*(inds[j] < self.dims[j])
        return vals[keeps], np.ravel_multi_index(inds,self.dims,mode="clip")[keeps]
