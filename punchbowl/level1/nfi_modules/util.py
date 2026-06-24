import os
import sys
import copy
import pickle
import resource
import traceback
from functools import wraps
from multiprocessing import Queue, Process

import numba
import numpy as np
from scipy.io import readsav


def processify(func):
    '''Decorator to run a function as a process.
    Be sure that every argument and the return value
    is *pickable*.
    The created process is joined, so the code does not
    run in parallel.
    '''

    def process_func(q, *args, **kwargs):
        try:
            ret = func(*args, **kwargs)
        except Exception:
            ex_type, ex_value, tb = sys.exc_info()
            error = ex_type, ex_value, ''.join(traceback.format_tb(tb))
            ret = None
        else:
            error = None

        q.put((ret, error))

    # register original function with different name
    # in sys.modules so it is pickable
    process_func.__name__ = func.__name__ + 'processify_func'
    setattr(sys.modules[__name__], process_func.__name__, process_func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        q = Queue()
        p = Process(target=process_func, args=[q] + list(args), kwargs=kwargs)
        p.start()
        ret, error = q.get()
        #p.join()

        if error:
            ex_type, ex_value, tb_str = error
            message = '%s (in subprocess)\n%s' % (ex_value.message, tb_str)
            raise ex_type(message)

        return ret
    return wrapper

def masked_median_filter(data,mask,radius,footprint=None,missing=0.0,footprint_ind_offset=0):

    if(footprint is None):
        if(np.isscalar(radius)): radarr = np.array([radius]*data.ndim)
        else: radarr = radius
        coordas = np.indices(2*radarr+1)
        for i in range(0,len(radarr)): coordas[i] = (coordas[i]-radarr[i])/radarr[i]
        radii = np.sum(coordas**2,axis=0)**0.5
        footprint = radii <= 1

    flatinds = np.ravel_multi_index(np.indices(data.shape),data.shape).flatten()
    footprint_inds = np.ravel_multi_index(np.indices(footprint.shape),footprint.shape)
    footprint_inds = np.array(np.unravel_index(footprint_inds[footprint],footprint.shape))
    footprint_inds = footprint_inds.transpose(np.roll(np.arange(footprint_inds.ndim),-1))
    data_filt = np.zeros(data.size)+missing
    footprint_pad = np.floor(0.5*np.array(footprint.shape)).astype(np.int32)
    footprint_pad = np.array([footprint_pad,footprint_pad]).T

    data_fppad = np.pad(data,footprint_pad)
    dat_pad_shape = data_fppad.shape
    data_fppad = data_fppad.flatten()
    mask_fppad = np.pad(mask,footprint_pad).flatten()
    tparg = np.roll(np.arange(footprint_inds.ndim),1)
    data_filt = _masked_medfilt_inner(flatinds,data,footprint_inds,footprint_ind_offset,tparg,dat_pad_shape,data_fppad,mask_fppad,data_filt)
    #for ind in flatinds:
    #    ijkpad = np.unravel_index(ind,data.shape)+footprint_inds-footprint_ind_offset
    #    ijkpad = np.ravel_multi_index(ijkpad.transpose(tparg),dat_pad_shape)
    #    dat = data_fppad[ijkpad]
    #    good = mask_fppad[ijkpad]
    #    if(np.sum(good) > 0): data_filt[ind] = np.median(dat[good])
    return data_filt.reshape(data.shape)

@numba.jit(fastmath=True, parallel=True, forceobj=True)
def _masked_medfilt_inner(flatinds,data,footprint_inds,footprint_ind_offset,tparg,dat_pad_shape,data_fppad,mask_fppad,data_filt):
    for ind in flatinds:
        ijkpad = np.unravel_index(ind,data.shape)+footprint_inds-footprint_ind_offset
        ijkpad = np.ravel_multi_index(ijkpad.transpose(tparg),dat_pad_shape)
        dat = data_fppad[ijkpad]
        good = mask_fppad[ijkpad]
        if(np.sum(good) > 0): data_filt[ind] = np.median(dat[good])
    return data_filt

# This routine multiplies a matrix with each element of a set of vectors.
# The vectors must be numpy arrays dimensioned nvec by ndim, where ndim is the
# dimensionality of the space, while the matrix is ndim by ndim. This applies in
# general for other operations of a set of vectors with a single vector --
# i.e., the vector space index must be the last one. For
# instance, to add a vector shift to a set of vectors, the array of
# vectors must also be nvec by ndim. Then they can be added as c = a+b. See
# https://numpy.org/doc/stable/user/basics.broadcasting.html
# This is somewhat backward to how dot products otherwise work in numpy
# So the order of operations has to be reversed compared to normal and the
# matrix must be transposed. i.e., instead of v2 = np.dot(fwd,v1), it's necessary
# to instead do v2 = np.dot(v1,fwd.T). There's may be a built in numpy way to
# do this, but I haven't found it so far. linalg.multi_dot doesn't appear to
# function any differently from dot in this case. I've implemented it
# here as a very small subroutine rather than spreading it all over the code
# for ease of maintenance and explanation. It works for single vectors, too.
def multivec_matmul(a,b): return np.dot(b,a.T)

# Forward rolling transpose, for switching from coordinate dimension last to
# coordinate dimension first in multidimensional coordinate arrays:
def ftp(a):
	return a.transpose(np.roll(np.arange(a.ndim),1))

# Backward rolling transpose, for switching from coordinate dimension first to
# coordinate dimension last in multidimensional coordinate arrays:
def btp(a):
	return a.transpose(np.roll(np.arange(a.ndim),-1))

# Numpy's indices method is extremely useful, but it puts the coordinate
# dimension (e.g., ijk) first, but for easy vector operations it should be last.
# Transposing puts the coordinate dimension last but it also reverses all of
# the other dimensions, which gets super confusing. This does a `roll' transpose
# which just shifts the dimensions forward by 1. Very simple, but you can see how
# it could get unggkljhly real quick
def rindices(dims,**kwargs):
	ia = np.indices(dims,**kwargs)
	return btp(ia)

def bindown(d,n):
    inds = np.ravel_multi_index(np.floor((np.indices(d.shape).T*n/np.array(d.shape))).T.astype(np.uint32),n)
    return np.bincount(inds.flatten(),weights=d.flatten(),minlength=np.prod(n)).reshape(n)

def bindown2(d,f):
    n = np.round(np.array(d.shape)/f).astype(np.int32)
    inds = np.ravel_multi_index(np.floor((np.indices(d.shape).T*n/np.array(d.shape))).T.astype(np.uint32),n)
    return np.bincount(inds.flatten(),weights=d.flatten(),minlength=np.prod(n)).reshape(n)

def binup(d,f):
    n = np.round(np.array(d.shape)*np.round(f)).astype(np.int32)
    inds = np.ravel_multi_index(np.floor((np.indices(n).T/np.array(f))).T.astype(np.uint32),d.shape)
    return np.reshape(d.flatten()[inds],n)

def as_dict(rec):
    """ turn a numpy recarray record into a dict. this is mostly useful
    just to have a human readable output of a record on the console.

    as_dict(my_data[234])
    """

    return {name:rec[name] for name in rec.dtype.names}

@processify
def get_iris_data(iris_filename,key=None):
    stuff = readsav(iris_filename,python_dict=True)
    if(key is None): return stuff.keys()
    else: return as_dict(stuff[key])

def get_mask_errs(dat_cube, iris_err_fac, error_cube=None, filt_thold=2.5):
    dat_arr = copy.deepcopy(dat_cube)
    dat_filt = masked_median_filter(dat_arr,(np.isnan(dat_arr)==0),np.array([1,2,1]))
    if(error_cube is None): error_cube = ((iris_err_fac**2+np.clip(dat_arr,0,None)*iris_err_fac)**0.5).astype('float64')

    dat_median = np.nanmedian(np.abs(dat_filt))
    dat_mask = (np.isnan(dat_arr) + (dat_arr < 0.00*error_cube) + np.isnan(error_cube) +
                  (np.abs(dat_arr-dat_filt) > filt_thold*(dat_median+np.abs(dat_filt)))) > 0

    return dat_mask, error_cube
