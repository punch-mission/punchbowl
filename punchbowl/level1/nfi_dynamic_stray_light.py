import os, copy, importlib, numpy as np, matplotlib.pyplot as plt
from sys import path
from astropy.time import Time
from astropy.io import fits

from astropy.wcs import WCS, FITSFixedWarning
from scipy.ndimage import gaussian_filter
plt.rcParams.update({'image.origin':'lower'})

file_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(file_dir)
module_path = os.path.join(src_dir,'nfi_modules')
path.append(module_path)

# native libraries
import nfi_modules.fwdmats, nfi_modules.reconstruct
importlib.reload(nfi_modules.fwdmats)
importlib.reload(nfi_modules.reconstruct)

from nfi_modules.reconstruct import reconstruct_nfi_straylight
from nfi_modules.fwdmats import generate_nfi_fwdmats
from nfi_modules.util import bindown


def remove_nfi_stray_light():
    pass