"""
===================
Decoding square root encoded data
===================

Data downlinked from the individual PUNCH spacecraft are usually square root encoded. This decoding is completed as part of the regular data processing pipeline for end-user data products. If using Level 0 manually, you may want to decode this data manually, as outlined here.
"""

# %%
# Data downlinked from the individual PUNCH spacecraft are usually square root encoded. This decoding is completed as part of the regular data processing pipeline for end-user data products. If using Level 0 manually, you may want to decode this data manually, as outlined here.

# %%
# Load libraries

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from punchbowl.data import load_ndcube_from_fits
from punchbowl.data.sample import PUNCH_DK4
from punchbowl.data.visualize import cmap_punch
from punchbowl.level1.sqrt import decode_sqrt

# %%
# We can begin by loading sample data into an NDCube object. Here we'll load a sample level 0 LED dark image from NFI. Note that if running with downloaded PUNCH data, replace the sample PUNCH_DK4 data below with a path pointing towards a FITS file.

# %%
cube = load_ndcube_from_fits(PUNCH_DK4)

# %%
# Note that for level 0 data the data is still square-root encoded, and will be unpacked in subsequent levels. You can manually decoded this data with the same pipeline function.

# %%
data_decoded = decode_sqrt(cube.data,
                      from_bits = 16,
                      to_bits = 11,
                      ccd_gain_top = 4.94,
                      ccd_gain_bottom = 4.89,
                      ccd_offset = 400,
                      ccd_read_noise = 17
                    )

# %%
# Now that we have this data square-root decoded, we can plot the image.

# %%
fig, ax = plt.subplots(figsize=(9.5, 7.5), subplot_kw={"projection":cube.wcs})

im = ax.imshow(data_decoded, cmap=cmap_punch, norm=LogNorm(vmax=450))

lon, lat = ax.coords
lat.set_major_formatter("dd")
lon.set_major_formatter("dd")
ax.set_facecolor("black")
ax.coords.grid(color="white", alpha=.25, ls="dotted")
ax.set_xlabel("Helioprojective longitude")
ax.set_ylabel("Helioprojective latitude")
ax.set_title("PUNCH Level 0 LED Image")
fig.colorbar(im, ax=ax, label="DN")
plt.show()

# %%
