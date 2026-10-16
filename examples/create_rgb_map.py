"""
=======================
Creating PUNCH RGB maps
=======================

This example creates RGB maps using PUNCH tB and pB sets of data.
These data are mapped to MZP triplets, which are then mapped to RGB components based on the formalism presented in DeForest et al. (2022) and Patel et al. (2023).
"""

#%%
import matplotlib.pyplot as plt

from punchbowl.data.punch_io import load_ndcube_from_fits
from punchbowl.data.sample import PUNCH_PAM
from punchbowl.data.visualize import generate_tbpb_to_rgb_map

#%%
# We'll being by loading data from a sample PAM data product into an ndcube object.
# Note that other polarized data could be substituted in place - see the example notebook on querying data using Fido.
punch_cube = load_ndcube_from_fits(PUNCH_PAM)


#%%
# Use the punchbowl function to map PUNCH polarization channels to RGB colorspace.
# Parameters can be tuned to enhance the visual appearance as required.
rgb_sat, rgb_raw = generate_tbpb_to_rgb_map(punch_cube,
                                            gamma=1/2.2,
                                            frac=.1,
                                            s_boost=1.25)

#%%
# Display the result image.
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection=punch_cube.wcs.slice(2,2))
ax.imshow(rgb_sat, origin='lower')
ax.set_axis_off()
ax.text(0.005,0.005,f"{punch_cube.meta.datetime.strftime('%Y/%m/%d %H:%M:%S' + 'UT')}",
        horizontalalignment='left', transform=ax.transAxes,
        fontsize=12, color='white')
