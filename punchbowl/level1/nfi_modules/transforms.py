# This is a minimal implementation of the coordinate transform used by get_sparse_response_matrix.
# All it does is compare the 'names' attributes of the input coords to that of the output coords.
# This works for identical coordinates, rearrangements, and downprojections, but anything
# more complex will get zeroed out. In terms of the necessary pieces for instantiating
# and applying it, though, everything should work the same from the outside and it can be 
# subclassed to add the features for any desired transformation. Note that
# these coordinate transforms might NOT be invertible in general (e.g., projection)
# unlike the transforms in the coord_grid class.

import numpy as np
from nfi_modules.util import multivec_matmul

class coord_transform:
    # Setup: takes two objects defining the two coordinate system. The minimal implementation
    # just compares the names attributes. This can be changed by overriding the
    # init_transform() method. Similarly, the transform itself is taken to be
    # an affine transform, but this can be changed by overriding the transform method.
    def __init__(self,coords_in,coords_out):
        [self.coords_in,self.coords_out] = [coords_in, coords_out]
        self.init_transform()
    def init_transform(self):
        [ndin,ndout] = [len(self.coords_in.frame.names), len(self.coords_out.frame.names)]
        [self.origin,self.fwd] = [np.zeros(ndin), np.zeros([ndout,ndin])]
        for i in range(0,ndout):
            for j in range(0,ndin): self.fwd[i,j] = self.coords_out.frame.names[i] == self.coords_in.frame.names[j]
    # As currently written will only work on one coordinate point at at time...
    def transform(self,coords): return multivec_matmul(self.fwd,coords)+self.origin

# A trivial coordinate frame object for use by the minimal implementation of the coord_transform
# class. Only contents are a set of names for the coordinates.
class trivialframe: 
    def __init__(self,names): self.names=names
