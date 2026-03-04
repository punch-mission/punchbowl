"""
===================
Animating PUNCH data
===================

How to animate PUNCH data using built-in plotting tools
"""

# %%  [markdown]
# First we'll load a set of libraries. This is minimal, but will give us the tools to query a sample of data to download and animate.

from datetime import datetime

from sunpy.net import Fido, attrs

from punchbowl.data.visualize import animate_punch

# %% [markdown]
# Next we'll query a sample of data to animate. Note that you can modify the time range, the product code, and the data version code.
# Two useful datasets to visualize are CAM and PAM - level 3 clear and polarized low-noise mosaics.

# %%
start_time, end_time = datetime(2025, 10, 31, 0, 0, 0), datetime(2025, 10, 31, 2, 0, 0)
product_code = "L3_CAM"    # Or L3_PAM for polarized data
data_type = "Unpolarized"  # Or "Polar_BpB" for polarized data
version_code = "v0j"

result = Fido.search(
                attrs.Time(start_time, end_time),
                attrs.Source("PUNCH-MOSAIC"),
                attrs.Instrument(data_type)
            )

result = result[0][[i for i, r in enumerate(result[0]['fileid']) if product_code in r and version_code in r]]

result

# %% [markdown]
# The resulting files can then be downloaded.

# %%
files = Fido.fetch(result)

# %% [markdown]
# That file list is then passed into the animator function, along with an output filename.

# %%
animate_punch(files, output_path="PUNCH_CAM.mp4")

# %% [markdown]
# .. raw:: html
#
#    <video width="100%" controls>
#      <source src="PUNCH_CAM.mp4" type="video/mp4">
#    </video>
