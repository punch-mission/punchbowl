import time

import numpy as np
from scipy.sparse import csc_matrix

def element_source_responses(source,
                             detector,
                             transform,
                             nbuf = 10**7,
                             dtype="float32"):
    """
    Generate the sparse response matrix for the general linear forward problem,
    mediated by a pair of coordinate systems and a coordinate transform.
    Has the following arguments:

        source:    An object which defines a set of indexed source, or basis,
                   elements. The element_grid class is the exemplar for these objects.
                   They must implement an elements method as well as
                   nelm, nadr, and coords attributes with usage consistent with their
                   definitions in element_grid.
        detector:  An object which defines a set of detector elements. The element_grid class
                   is the exemplar for these objects. They must implement a response method
                   as well as nelm and coords attributes with usage consistent with their
                   definitions in element_grid.
        transform: A function which returns an object that maps points in the source
                   coordinate system to the detector coordinate system. This coordinate
                   transform does not need to be reversible (e.g., the source system
                   can be 3D and the detector system can be 2D) -- only the forward
                   direction must be well defined. Naturally, this may result in a
                   singular response matrix. Transform must be callable with the
                   following syntax:
                       transformer = transform(source.coords,detector.coords)
                   The transformer object it returns must be callable with the
                   following syntax:
                       pts_det_frame = transformer(pts_src_frame)
                   where pts_det_frame are pts_src_frame transformed from
                   the source frame to the detector frame
    There is a close mathematical relationship between source and detector
    elements -- almost all of their implementation code in element_grid is
    shared, and if they're swapped in the call to this routine, the result
    should merely be the transpose of the response matrix (untested).
    """
    # Create the coordinate transformation object:
    transformer = transform(source.coords,detector.coords)
    shape = (detector.nelm,source.nelm) # Number of input/outputs
    

    # Updating the sparse matrix comes with some overhead. We use buffers so we
    # don't have to do it so often:
    [ibuf_in,ibuf_out] = [np.zeros(nbuf,dtype=np.uint32),np.zeros(nbuf,dtype=np.uint32)]
    valbuf = np.zeros(nbuf,dtype=dtype)

    amat = csc_matrix(shape,dtype=dtype) # The initial empty sparse matrix
    [icur, t0] = [0, time.time()] # Buffer position and starting time (for printing status)
    for i in range(source.nadr): # Loop over each source address
        # Get the source elements for the current address:
        [src_elms,src_vals,src_pnts] = source.elements(i)

        for j in range(len(src_vals)): # Loop over each element for the current address:
            # Compute the detector response to this coordinate point:
            [det_vals,det_elms] = detector.response(transformer.transform(src_pnts[j]))

            # If the buffer is full, update the sparse matrix and reset the buffer position:
            if icur+len(det_elms) >= nbuf:
                amat += csc_matrix((valbuf[0:icur],(ibuf_out[0:icur],ibuf_in[0:icur])),shape=shape,dtype=dtype)
                amat.sum_duplicates()
                icur = 0

            # Put the current values in the buffer and update the buffer position:
            ibuf_in[icur:icur+len(det_elms)] = src_elms[j]
            ibuf_out[icur:icur+len(det_elms)] = det_elms
            valbuf[icur:icur+len(det_elms)] = det_vals*src_vals[j]/np.prod(detector.nsubgrid)/np.prod(source.nsubgrid)
            icur+=len(det_elms)

    # Update the sparse matrix with the last value in the buffer
    if(icur > 0):
        amat += csc_matrix((valbuf[0:icur],(ibuf_out[0:icur],ibuf_in[0:icur])),shape=shape,dtype=dtype)
    amat.sum_duplicates() # Clean up the sparse matrix
    return amat.tocsc() # Done.
