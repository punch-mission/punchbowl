#%%
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from sunpy.net import Fido, attrs

from punchbowl.data.punch_io import load_ndcube_from_fits
from punchbowl.data.visualize import generate_mzp_to_rgb_map

#%% md
# Querying some PUNCH polarized data
#%%
# This is the only line you change for different queries
start_time, end_time, product_code = datetime(2025, 11, 1), datetime(2025, 11, 2), "L3_PIM"

result = Fido.search(
                attrs.Time(start_time, end_time),
                attrs.Source("PUNCH-MOSAIC"),
                attrs.Instrument("Polar_MZP")
            )
result = result[0][[i for i, r in enumerate(result[0]['fileid']) if product_code in r]]
#%%
# From all the queried outputs, we download one file to show an example usage.
punch_file = Fido.fetch(result[101])
#%%
# Load data from fits to ndcube format using inbuilt punchbowl function.
punch_cube = load_ndcube_from_fits(punch_file[0])
#%%
# Check if the data contains any infs and mask them out.
parr = np.zeros((3,4096,4096))
mask = np.isfinite(punch_cube.data)
parr[mask] = punch_cube.data[mask]
#%%
# Use the punchbowl function to map PUNCH polarizer triplets to RGB colorspace.
# Parameters can be tuned to enhance the visual appearance as required.
rgb_sat, rgb_raw = generate_mzp_to_rgb_map(punch_cube.data,
                                        gamma=0.6,
                                        frac=.15,
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
#%%
