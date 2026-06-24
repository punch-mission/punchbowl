import os
import copy
import importlib
from sys import path

import astropy.wcs
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning
from scipy.ndimage import gaussian_filter

plt.rcParams.update({'image.origin':'lower'})

file_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(file_dir)
module_path = os.path.join(src_dir,'nfi_modules')
path.append(module_path)

# native libraries
import nfi_modules.fwdmats
import nfi_modules.reconstruct

importlib.reload(nfi_modules.fwdmats)
importlib.reload(nfi_modules.reconstruct)

from nfi_modules.fwdmats import generate_nfi_fwdmats
from nfi_modules.reconstruct import reconstruct_nfi_straylight
from nfi_modules.util import bindown

# punchbowl libraries
from punchbowl.data.punchcube import PUNCHCube


def get_center(crval,cdelt,bin_factor):
    return (-crval/cdelt)/bin_factor

def get_fwd_mat_inputs(data: PUNCHCube,
                       bin_factor):
    data_wcs = data.wcs
    data_meta = data.meta

    xcens = get_center(data_wcs.wcs.crval[0],data_wcs.wcs.cdelt[0],bin_factor)
    ycens = get_center(data_wcs.wcs.crval[1],data_wcs.wcs.cdelt[1],bin_factor)

    crots = data_meta['CROTA'].value*np.pi/180

    return xcens, ycens, crots


def remove_nfi_stray_light(data: PUNCHCube,
                           bin_factor: int = 4):
    xcens, ycens, crots = get_fwd_mat_inputs(data=data,bin_factor=bin_factor)

    return
