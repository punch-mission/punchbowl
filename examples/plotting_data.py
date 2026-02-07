"""
===================
Plotting PUNCH data
===================

How to plot PUNCH data using built-in plotting tools
"""

# %%  [markdown]
# So you've downloaded a PUNCH FITS file, and you want to display it in Python in a simple way. We've encapsulated a flexible plotting function within the punchbowl repository that users can use and build on.

# %%  [markdown]
# First we'll load a set of libraries. This is minimal, but will provide a sample file to use for plotting, and the plotting function itself.

# %%
from punchbowl.data.sample import PUNCH_CAM
from punchbowl.data.visualize import plot_punch

# %%
# Next we'll plot the data. Note that you can provide as input either a path to a FITS file directly, or alternatively a loaded NDCube object.

# %%
plt, ax = plot_punch(PUNCH_CAM)

# %%  [markdown]
# This is using the default parameters, but the function is flexible to plot in a few different ways. For example, let's say we want a minimal plot of just the data, but still with some annotation.

# %%
plt, ax = plot_punch(PUNCH_CAM, axes_off=True)

# %%  [markdown]
# For a more complex example let's bring down the vmax value a bit, relabel the axes, and turn off the colorbar.

# %%
plt, ax = plot_punch(PUNCH_CAM, vmax=5e-13, axes_labels=(r'$X_{im}$', r'$Y_{im}$'), colorbar=False)
# %%

# %%  [markdown]
# A variety of other plotting elements and options can be controlled - for full details see the function docstring or ask the SOC.
