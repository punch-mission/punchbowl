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

from punchbowl.data.sample import PUNCH_CAM_SEQUENCE
from punchbowl.data.visualize import animate_punch

# %% [markdown]
# Let's pass the sample data sequence into the animator function, along with an output filename. Note that ordinarily you could pass in a list of input files to animate.

# %%
animate_punch(PUNCH_CAM_SEQUENCE, output_path="PUNCH_CAM.mp4", n_jobs=1,
              axes_off=True, trim_edge=(0.13, 0.68))

# %% [markdown]
# .. raw:: html
#
#    <video width="100%" controls>
#      <source src="PUNCH_CAM.mp4" type="video/mp4">
#    </video>
