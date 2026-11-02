"""
==============================
Colormaps for PUNCH Data
==============================

How to plot PUNCH data and derived quantities using built-in colormaps
"""

# %%
# Let's take a look at how to plot some PUNCH data and derived quantities using the built-in colormaps.


# %%
# Load libraries

import numpy as np

import punchbowl.data.color  # Note that this import is needed to register PUNCH colormaps with matplotlib
from punchbowl.data import punch_io
from punchbowl.data.punchcube import PUNCHCube
from punchbowl.data.sample import PUNCH_PAM
from punchbowl.data.visualize import plot_punch

# %%
# Let's start by loading a PUNCH polarized low-noise mosaic into a datacube.

# %%
datacube = punch_io.load_ndcube_from_fits(PUNCH_PAM)

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

fig, ax = plot_punch(polarization_degree, title_prefix="PUNCH Degree of Polarization", cmap="punch_p",
                     vmin=0, vmax=1, gamma=1, colorbar_label="Degree of polarization")

# %%
# In the same way we can also compute tau.

# %%
tau = PUNCHCube(data = np.arcsin(np.sqrt((1 - polarization_degree.data) / (1 + polarization_degree.data))),
                wcs = datacube.wcs[0],
                meta = datacube.meta)

fig, ax = plot_punch(tau, title_prefix="PUNCH Tau", cmap="punch_tau",
                     vmin=0, vmax=np.pi/2, gamma=1, colorbar_label="tau")
