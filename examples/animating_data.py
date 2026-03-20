"""
===================
Animating PUNCH data
===================

How to animate PUNCH data using built-in plotting tools
"""

# %%  [markdown]
# First we'll load a set of libraries. This is minimal, but will give us the tools to query a sample of data to download and animate.
# Note that for animation, you'll need a local copy of ffmpeg, which can be installed through tools such as homebrew or conda. Depending on your environment you may also need to install a corresponding python package with a command such as ``pip install ffmpeg-python``.
# Also note that punchbowl is in active development at the moment. To install the bleeding-edge version, use a command such as ``pip install git+https://github.com/punch-mission/punchbowl@main``.

from sunpy.net import Fido
from sunpy.net import attrs as a

import punchbowl  # Note that this import is needed to register PUNCH fido tools
from punchbowl.data.visualize import animate_punch

# %% [markdown]
# Next we'll query a sample of data to animate. Note that you can modify the time range, the product code, and the data version code.
# Two useful datasets to visualize are CAM and PAM - level 3 clear and polarized low-noise mosaics.

# %%
result = Fido.search(a.Time('2025/10/14 14:30:00', '2025/10/14 17:00:00'),
                     a.punch.ProductCode.ca, # (ca for clear low-noise), or pa for polarized low-noise, etc.
                     a.Instrument.m, # (m for mosaic), or a.Instrument.nfi_4, etc for earlier levels.
                     a.Level.three,
                     a.punch.DataVersion.newest, # or a.punch.DataVersion.zero_j, etc.
                     a.punch.FileType.fits) # or a.punch.FileType.jp2

result

# %% [markdown]
# The resulting files can then be downloaded.
# Note that occasionally data access through these tools can be down for maintenance or other issues. To instead query a list of local files you may have on hand, import the glob package into python, and then use a command such as ``files = glob.glob("path/to/files/*.fits")`` followed by ``files.sort()``.

# %%
files = Fido.fetch(result)

# %% [markdown]
# That file list is then passed into the animator function, along with an output filename.

# %%
animate_punch(files, output_path="PUNCH_CAM.mp4", axes_off=True, trim_edge=(0.13, 0.68))

# %% [markdown]
# .. raw:: html
#
#    <video width="100%" controls>
#      <source src="PUNCH_CAM.mp4" type="video/mp4">
#    </video>
