"""
===================
Querying PUNCH Data
===================

A notebook guide to querying and loading PUNCH data using SunPy.
"""

# %%
# This notebook provides a guide on how to use tools to query PUNCH data from the VSO / SDAC using Python tools, and how to load and display this data using SunPy.


# %%
# Load libraries

from sunpy.map import Map
from sunpy.net import Fido
from sunpy.net import attrs as a

from punchbowl.data.visualize import cmap_punch

# %%
# Data querying
# ------------
#
# With a range of dates and a PUNCH instrument in mind, we can begin querying data. Here we'll consider data from the first hour of the first of June, and focus on data from WFI-2 only.
#
# We can construct a query using the Fido tool, specifying search attributes:

# %%
time_range = a.Time('2025/06/01 00:00:00', '2025/06/01 01:00:00')
result = Fido.search(time_range, a.Instrument('WFI-2'))
result

# %%
# This results in a table of available data products that match the search criteria.
# Next, let's download the first file from this list of results:

# %%
files = Fido.fetch(result)

# %%
# This returns a list of paths to files that have been downloaded. Note that the Fido.fetch tool can specify a particular download directory for larger data searches.
# With that file downloaded, we can load it into a SunPy map object, and display it.

# %%
map = Map(files[0])
map.peek(cmap=cmap_punch)

# %%
# And that's it! From here the data is encapsulated into a SunPy map object, which supports that framework for plotting, coordinate transformations, etc. Note that in SunPy 7.0+, SunPy will use the PUNCH colormap natively, and will no longer need to be specified manually.
#
# Of course this is just one path, you could always load the data using Astropy fits tools, load it into an NDCube, or any other FITS-compliant tool.
