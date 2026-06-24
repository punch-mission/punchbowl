#%%
import os
import copy
import importlib
from sys import path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning
from scipy.ndimage import gaussian_filter

plt.rcParams.update({'image.origin':'lower'})

src_dir = os.path.dirname(os.path.abspath(__file__))
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

#punchbowl specific modules
from punchbowl.util import load_image_task


#%%
def nfi_core_flow(input_data: list[str],
                  ):
    """Flow pipeline for NFI (stray light processing)
        (Input FITS file; output PUNCH ndcube)"""

    # TODO: logger

    output_data = []

    for i, this_data in enumerate(input_data):
        data = load_image_task(this_data) if isinstance(this_data,str) else this_data
