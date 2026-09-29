"""
==============================
Plotting PUNCH data with color
==============================

How to plot PUNCH data and derived quantities using built-in colormaps
"""

# %%
# Let's take a look at how to plot some PUNCH data and derived quantities using the built-in colormaps.


# %%
# Load libraries

import numpy as np
from sunpy.net import Fido
from sunpy.net import attrs as a

import punchbowl  # Note that this import is needed to register PUNCH fido tools
import punchbowl.data.color  # Note that this import is needed to register PUNCH colormaps with matplotlib
from punchbowl.data import punch_io
from punchbowl.data.punchcube import PUNCHCube
from punchbowl.data.visualize import plot_punch

# %%
# Let's start with a search of some suitable data to plot.

# %%
result = Fido.search(a.Time('2025/08/31 20:00:00', '2025/08/31 21:00:00'),
                     a.punch.ProductCode.pa, # (ca for clear low-noise), or pa for polarized low-noise, etc.
                     a.Instrument.m, # (m for mosaic), or a.Instrument.nfi_4, etc for earlier levels.
                     a.Level.three,
                     a.punch.DataVersion.newest, # or a.punch.DataVersion.zero_j, etc.
                     a.punch.FileType.fits) # or a.punch.FileType.jp2

result

# %%
# This results in a table of available data products that match the search criteria.
# Next, let's download the first file from this list of results:

# %%
try:
    files = Fido.fetch(result[0][0])
except IndexError:
    print("Oops no files were found!")
    files = None
# %%
# With that file downloaded, let's load the file into memory.

# %%
if files:
    datacube = punch_io.load_ndcube_from_fits(files[0])

# %%
# Now we can plot the total brightness layer of this data. Note that it will use the default total brightness colortable.

# %%
fig, ax = plot_punch(datacube, layer=0)

# %%
# Now we can plot the polarized brightness layer of this data. Note that the punch plotter will automatically use the correct colormap here. You can also manually specify it by passing through cmap="punch_pb"

# %%
fig, ax = plot_punch(datacube, layer=1, title_prefix="PUNCH PAM pB", vmin=1e-15, vmax=1e-13)

# %%
# We can calculate derived quantities, such as the degree of polarzation. We'll compute that manually into an array, and then create a PUNCHCube object we can use to plot this.

# %%
polarization_degree = PUNCHCube(data = datacube.data[1,...] / datacube.data[0,...],
                                  wcs = datacube.wcs[0], # Taking a slice of the original WCS to make it 2D
                                  meta = datacube.meta)

fig, ax = plot_punch(polarization_degree, title_prefix="PUNCH Degree of Polarization", cmap="punch_p", vmin=0, vmax=1, gamma=1)

# %%
# In the same way we can also compute tau.

# %%
tau = PUNCHCube(data = np.arcsin(np.sqrt((1 - polarization_degree.data) / (1 + polarization_degree.data))),
                wcs = datacube.wcs[0],
                meta = datacube.meta)

fig, ax = plot_punch(tau, title_prefix="PUNCH Tau", cmap="punch_tau", vmin=0, vmax=np.pi/2, gamma=1)
