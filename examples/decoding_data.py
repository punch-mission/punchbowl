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

from punchbowl.data import load_ndcube_from_fits
from punchbowl.data.sample import PUNCH_PAM  # TODO - add L0 to sample data?
from punchbowl.level1.sqrt import decode_sqrt

# %%
path = "/Users/clowder/Downloads/PUNCH_L0_DK3_20250407194626_v1.fits"
cube = load_ndcube_from_fits(path)

# %%
decoded = decode_sqrt(cube.data,
                      from_bits = 22,
                      to_bits = 11,
                      scaling_factor = 64,
                      ccd_gain_right = 4.94,
                      ccd_gain_left = 4.89,
                      ccd_offset = 400,
                      ccd_read_noise = 17,
                    )
