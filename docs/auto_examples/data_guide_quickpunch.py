"""
========================
Guide to QuickPUNCH Data
========================

A notebook guide to working with QuickPUNCH data in Python
"""

# %%
# Load libraries

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from ndcube import NDCube
from sunpy.map import Map

from punchbowl.data.sample import QUICKPUNCH_NQN, QUICKPUNCH_WQM

# %%
# Open the HDU list, and read out the appropriate data
# As the data is RICE compressed, the *second* HDU contains the main data frame
# The third HDU contains a corresponding uncertainty array

with fits.open(QUICKPUNCH_WQM) as hdul:
    print("WFI QuickPUNCH HDU List:")
    hdul.info()
    wfi_qp_data = hdul[1].data
    wfi_qp_header = hdul[1].header
    wfi_qp_uncertainty = hdul[2].data

with fits.open(QUICKPUNCH_NQN) as hdul:
    print("NFI QuickPUNCH HDU List:")
    hdul.info()
    nfi_qp_data = hdul[1].data
    nfi_qp_header = hdul[1].header
    nfi_qp_uncertainty = hdul[2].data

# %%
# The primary data arrays are stored as standard ndarrays
# The uncertainty data array has the dimensions as the primary data array
# Both the primary and uncertainty data arrays share the same header, contained in the primary HDU

print("WFI data array size:", wfi_qp_data.shape)
print("WFI uncertainty array size:", wfi_qp_uncertainty.shape)

# %%
# The corresponding headers can be queried as AstroPy header objects

wfi_qp_header["DATE-OBS"]

# %%
# The header information can be converted into an AstroPy WCS object

wfi_qp_data_wcs = WCS(wfi_qp_header)
nfi_qp_data_wcs = WCS(nfi_qp_header)

# %%
# Construct a SunPy Map object of out this data

wfi_qp_data_map = Map(wfi_qp_data, wfi_qp_header)
nfi_qp_data_map = Map(nfi_qp_data, nfi_qp_header)

# %%
# Display this SunPy Map object

wfi_qp_data_map

# %%
# Display this SunPy Map object

nfi_qp_data_map

# %%
# Construct an NDCube object out of this data

wfi_qp_data_ndcube = NDCube(wfi_qp_data, wcs=wfi_qp_data_wcs)
nfi_qp_data_ndcube = NDCube(nfi_qp_data, wcs=nfi_qp_data_wcs)

# %%
# Take a quick look at these NDCube objects

wfi_qp_data_ndcube, nfi_qp_data_ndcube

# %%
# Display this data in a regular plotting environment, using the associated WCS

plt.figure(figsize=(7.5, 7.5))
ax = plt.subplot(111, projection=wfi_qp_data_wcs)
plt.imshow(np.log(wfi_qp_data), cmap="Greys_r", vmin=-16, vmax=0)
lon, lat = ax.coords
lat.set_ticks(np.arange(-90, 90, 5) * u.degree)
lon.set_ticks(np.arange(-180, 180, 5) * u.degree)
lat.set_major_formatter("dd")
lon.set_major_formatter("dd")
ax.set_facecolor("black")
ax.coords.grid(color="white", alpha=.1)
plt.xlabel("Helioprojective longitude")
plt.ylabel("Helioprojective latitude")
plt.scatter(0, 0, s=240, color="k", transform=ax.get_transform("world"))
plt.title("QuickPUNCH Mosaic total brightness - " + wfi_qp_header["DATE-OBS"] + "UT")

# %%
# Display this data in a regular plotting environment, using the associated WCS

plt.figure(figsize=(7.5, 7.5))
ax = plt.subplot(111, projection=nfi_qp_data_wcs)
plt.imshow(np.log(nfi_qp_data), cmap="Greys_r", vmin=-16, vmax=0)
lon, lat = ax.coords
lat.set_ticks(np.arange(-90, 90, 5) * u.degree)
lon.set_ticks(np.arange(-180, 180, 5) * u.degree)
lat.set_major_formatter("dd")
lon.set_major_formatter("dd")
ax.set_facecolor("black")
ax.coords.grid(color="white", alpha=.1)
plt.xlabel("Helioprojective longitude")
plt.ylabel("Helioprojective latitude")
plt.scatter(0, 0, s=240, color="k", transform=ax.get_transform("world"))
plt.title("QuickPUNCH NFI total brightness - " + nfi_qp_header["DATE-OBS"] + "UT")

# %%
# Again noting that these files are compressed, additional keywords will be visible when viewing these FITS files outside of Python.
# These keywords relate to the compression implementation, and can be retrieved using astropy.io.fits, if needed, using the disable_image_compression keyword.

with fits.open(QUICKPUNCH_WQM, disable_image_compression=True) as hdul:
    header_compression = hdul[1].header

header_compression

# %%
