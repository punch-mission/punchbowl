"""
=====================
Creating PUNCH movies
=====================

This example creates movies off-screen.
It downloads the data based on your input of a date range, and required data level and data product.
The data products are documented at: https://punchbowl.readthedocs.io/en/latest/data/data_codes.html
"""

import os
import sys
import glob
from typing import List, Union, Optional
from pathlib import Path
from datetime import datetime

import astropy.units as u
import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display
from matplotlib.colors import LogNorm, PowerNorm
from parfive import Downloader, Results
from sunpy.map import Map
from sunpy.net import Fido, attrs
from sunpy.net.vso import VSOClient

from punchbowl.data.punch_io import load_ndcube_from_fits
from punchbowl.data.visualize import cmap_punch

# Set a specific height (e.g., 1000px)
display(HTML("<style>.jp-OutputArea-output {max-height: 1200px;}</style>"))

"""## Defining the data downloader class

This is an example of how a wrapper I wrote to access PUNCH data using `Fido` as the interfacing code to query the VSO and the SDAC where the PUNCH data are stored.


"""

class PUNCHDataDownloader:
    """
    Download PUNCH mission data products using SunPy's Fido interface.

    Author: Raphael Attie
    Acknowledgements: Marcus Hughes and Chris Lowder @PUNCH SOC
    """

    def __init__(
        self,
        source: str = "PUNCH-MOSAIC",
        instrument: str = "Unpolarized",
    ):
        """
        Initialize the PUNCH data downloader.

        Parameters
        ----------
        source : str, optional
            Data source identifier (default: "PUNCH-MOSAIC")
        instrument : str, optional
            Instrument name (default: "Unpolarized")
        """
        self.source = source
        self.instrument = instrument
        self.output_dir = None
        self.output_path = None

    def search(
        self,
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        product_code: str
    ) -> List[str]:
        """
        Search for PUNCH data files matching the specified criteria.

        Parameters
        ----------
        start_time : str or datetime
            Start time in format 'YYYY/MM/DD HH:MM:SS' or datetime object
        end_time : str or datetime
            End time in format 'YYYY/MM/DD HH:MM:SS' or datetime object
        product_code : str
            Product code identifier (e.g., 'L3_CIM', 'L2_CTM')

        Returns
        -------
        List[str]
            List of Fido URLs for matching data files
        """
        # Convert datetime objects to strings if needed
        if isinstance(start_time, datetime):
            start_time = start_time.strftime('%Y/%m/%d %H:%M:%S')
        if isinstance(end_time, datetime):
            end_time = end_time.strftime('%Y/%m/%d %H:%M:%S')

        print(f"Searching for PUNCH data...")
        print(f"  Time range: {start_time} to {end_time}")
        print(f"  Source: {self.source}")
        print(f"  Instrument: {self.instrument}\n")

        # Create time range attribute
        time_range = attrs.Time(start_time, end_time)

        # Search using Fido
        result = Fido.search(
            time_range,
            attrs.Source(self.source),
            attrs.Instrument(self.instrument)
        )

        # Filter by product code
        fido_urls = result[0][[product_code in fileid for fileid in result[0]['fileid']]]
        print(f"✓ Found {len(result[0])} total files")


        if not fido_urls:
            print(f"⚠️  No files found matching product code '{product_code}'")
            return []

        print(f"✓ Filtered to {len(fido_urls)} files matching product code '{product_code}'\n")

        return fido_urls

# %%
# Define your time range and data product code
# After execution, look at the summary of how many files matched the query.
# Product code are documented at https://punchbowl.readthedocs.io/en/latest/data/data_codes.html
#
# Currently, clear-filter data (unpolarized) are supported for this notebook.
# Enter start time and end time of the event
# For long time range, consider a lower cadence, e.g. skipping every 4 frames (skip = 4)
# That will keep the memory footprint manageable, and will mitigate visual flickering of the trefoil pattern.

start_time='2025/08/31 05:00:00'
end_time='2025/09/01 05:00:30' # Adding the last 30 seconds just make sure you get the one past 5:00 UTC
skip = 4  # Set `skip` to 0 for full cadence, but beware of disk space...

# %%
# After this session, try changing the times, and even look at different CMEs
# Be mindful of temporary disk space in your Google Colab (see next cell)
# PUNCH data are RICE-compressed: they take about 12 times less on disk than they do once in memory (RAM).
# One single PUNCH trefoil image takes ~10 MB when stored on disk. About 128 MB once decompressed in RAM.

product_code='L3_CIM'

# Naming convention for the output directory will use the start_time and end_time defined above.
stime = start_time.replace('/', '').replace(':', '').replace(' ', '_')
etime = end_time.replace('/', '').replace(':', '').replace(' ', '_')

# Output directory where all your downloaded data and PNG plots will land.
output_dir = Path('punch_data', stime +'_to_'+etime, product_code)

# %%
# [Optional] Setup data persistence by mounting your personal Google Drive
# By default, Google Colab gives you a temporary space where all your output are stored. Click on the folder icon in the side bar on the left of this window, it will reveal this temp space. You only have about 60 GB available. Whatever is written in that space is lost after the end of your Colab session (referred to as *Runtime* in Colab).  The commented code below, optionally, allows you to mount your own Google Drive for data persistence beyond this Runtime. It will prompt you to give access to your own Google Drive by Google Colab.
#
# After you have tried this with the temp space, you have the option to
# uncomment these 4 lines below if mounting your own Google Drive,
# ****Be sure to remove all leading spaces****
# If you do this, you will need to restart the run from this cell.

# from google.colab import drive
# drive.mount('/content/drive')
# gdrive_dir = Path('drive/MyDrive/Colab Notebooks/PUNCH Movies')
# output_dir = gdrive_dir / Path('punch_data', (stime +'_to_'+etime), product_code)

# %
# We first check how much data are available through VSO before starting any download.

dl = PUNCHDataDownloader()

urls = dl.search(
        start_time=start_time,
        end_time=end_time,
        product_code=product_code
)

"""**Executing the cell below should pop-up a window on the side bar about some Third-party widgets being used. Just close it.**"""

# %
# Download the FITS files with Fido, taking into account the `skip` parameter.
output_dir.mkdir(parents=True, exist_ok=True)
files = Fido.fetch(urls[::skip], path=output_dir)

# %
# A pop-up window should appear on the side bar about some Third-party widgets being used. Just close it.

# %
# Get list of local files, don't forget to sort them.
files = sorted(Path(output_dir).glob('*.fits'))
# Print a few filenames downloaded, just to sanity check it's what we expect
# You can check your downloaded FITS files on the side bar (folder icon), at the printed location.
files

# %
# Preparing the movie
# -------------------
# For making the movies, we use an interactive configurator with a downsampled series of frames to setup the intensity scaling. Once we are satisfied with the  intensity scaling, we move forward with printing the png frames off-screen, which will be given to ffmpeg for creating a movie file.


# basic binning to downsample the image for more efficiency with Google Colab and JavaScript display.
binning = 8  # 8 makes them 512x512, down from 4096 x 4096

# power-law for the intensity scaling (number advised by SOC)
gamma = 1/2.2

# Pre-load images and metadata, downsample images before adding to stack
cube_lr = []
metadata = []
for file in files:
    ndcube_obj = load_ndcube_from_fits(file)
    data = ndcube_obj.data[::binning, ::binning]
    data[data <0] = 0
    data = data**gamma
    cube_lr.append(data)
    metadata.append(ndcube_obj.meta)

# %
# Interactive plot
# ----------------

import numpy as np
from bokeh.io import output_notebook
from bokeh.layouts import column, row
from bokeh.models import (
    BasicTicker,
    ColorBar,
    ColumnDataSource,
    CustomJS,
    CustomJSTickFormatter,
    LinearColorMapper,
    LogColorMapper,
    LogTicker,
    Slider,
    Spacer,
)
from bokeh.plotting import figure, show

# %
# Enable Bokeh in Jupyter
output_notebook()

display(HTML("""
<style>
  /* JupyterLab output containers */
  .jp-OutputArea,
  .jp-OutputArea-child,
  .jp-OutputArea-output {
    max-height: none !important;
    height: auto !important;
    overflow: visible !important;
  }
</style>
"""))


def mpl_to_bokeh_palette(cmap, n=256):
    """Convert matplotlib colormap to Bokeh palette"""
    colors = cmap(np.linspace(0, 1, n))
    # Convert to hex colors
    palette = ['#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))
               for r, g, b, _ in colors]
    return palette

# %
# Convert your custom colormap
cmap_punch_bokeh = mpl_to_bokeh_palette(cmap_punch)


initial_image = cube_lr[0]

# %
# Create the figure
p = figure(frame_width=600, frame_height=600,
           x_range=(-180, 180), y_range=(-180, 180),
           title=f"Image 0 of {len(cube_lr)}")

# Add axis labels
p.xaxis.axis_label = "Apparent distance [Rs]"
p.yaxis.axis_label = "Apparent distance [Rs]"

# Create power-law color mapper

inv_gamma = 1.0/gamma

vmin = 1e-14**gamma
vmax = 1e-12**gamma
step_min = (2e-16**gamma)/10
step_max = (2e-14**gamma)/10

vmin_slider_min = (1e-14/2)**gamma
vmin_slider_max = (1e-14*2)**gamma
vmax_slider_min = (1e-12/2)**gamma
vmax_slider_max = (1e-12*2)**gamma

# Use a linear color mapper on the transformed data
color_mapper = LinearColorMapper(palette=cmap_punch_bokeh, low=vmin, high=vmax)


# Create a ColumnDataSource with the image data
source = ColumnDataSource(data={'image': [initial_image]})

# Display the image
p.image(image='image', x=-180, y=-180, dw=360, dh=360,
        color_mapper=color_mapper, source=source)


# Formatter that shows untransformed tick labels
colorbar_tick_formatter = CustomJSTickFormatter(args=dict(inv_gamma=inv_gamma), code="""
    // 'tick' is in transformed space; show original scale by inverting the power transform
    const original = Math.pow(tick, inv_gamma);
    // scientific notation with 2 significant digits
    return original.toExponential(2);
""")

# Create colorbar

color_bar = ColorBar(
    color_mapper=color_mapper,
    ticker=BasicTicker(),
    width=15,
    location=(0, 0),
    formatter=colorbar_tick_formatter
)

# Add colorbar to the figure
p.add_layout(color_bar, 'right')


### Create interactive sliders:

# Formatter for slider tick labels (invert back to original)
slider_tick_formatter = CustomJSTickFormatter(args=dict(inv_gamma=inv_gamma), code="""
    const original = Math.pow(tick, inv_gamma);
    return original.toExponential(2);
""")

vmin_slider = Slider(
    start=vmin_slider_min, end=vmin_slider_max, value=vmin, step=step_min,
    title="vmin: ",  # placeholder; we'll update via JS
    format=slider_tick_formatter, width=350
)
vmax_slider = Slider(
    start=vmax_slider_min, end=vmax_slider_max, value=vmax, step=step_max,
    title="vmax: ", format=slider_tick_formatter, width=350
)


# JavaScript callback to update color mapper range
color_callback = CustomJS(args=dict(color_mapper=color_mapper, vmin_slider=vmin_slider, vmax_slider=vmax_slider), code="""
    color_mapper.low = vmin_slider.value;
    color_mapper.high = vmax_slider.value;
""")

vmin_slider.js_on_change('value', color_callback)
vmax_slider.js_on_change('value', color_callback)


# image index slider (time dimension)
index_slider = Slider(start=0, end=len(cube_lr)-1, value=0, step=1, title="Image Index", width=690)

# 1) Prepare metadata for JS (one entry per image)
# Assuming you have a Python list `metadata` where each element has product_level, product_code,
# and slice.meta['DATE-OBS'] accessible as described.
js_metadata = []
for meta in metadata:
    # Extract only what you need and ensure it’s JSON-serializable (strings)
    date_obs = str(meta.get('DATE-OBS', ''))  # fallback empty string if missing
    js_metadata.append(dict(
        product_level=str(meta.product_level),
        product_code=str(meta.product_code),
        date_obs=date_obs
    ))

# Set initial title
m0 = js_metadata[0]
date0 = m0['date_obs'][:-4] if len(m0['date_obs']) >= 4 else m0['date_obs']
p.title.text = f"PUNCH Level-{m0['product_level']} {m0['product_code']} @ {date0} image 0 of {len(cube_lr)}"

# JavaScript callback to update the image
image_callback = CustomJS(args=dict(source=source, cube=cube_lr, p=p, n=len(cube_lr), meta_list=js_metadata), code="""
    const data = source.data;
    const idx = cb_obj.value;
    // Update image
    data['image'] = [cube[idx]];
    const meta = meta_list[idx] || {};
    const level = meta.product_level || '';
    const code = meta.product_code || '';
    const dateObs = (meta.date_obs || '');
    // Trim last 4 chars from DATE-OBS
    const dateTrimmed = dateObs.length >= 4 ? dateObs.slice(0, -4) : dateObs;
    const prefix = `PUNCH Level-${level} ${code}`;
    p.title.text = `${prefix} @ ${dateTrimmed} — Image ${idx} of ${n}`;
    source.change.emit();
""")

index_slider.js_on_change('value', image_callback)

# Create spacer for offset
spacer_vmin_slider = Spacer(width=20)
spacer_vmax_slider = Spacer(width=100)
spacer_index_slider = Spacer(width=30)
# Display the layout
show(column(row(spacer_vmin_slider , vmin_slider, vmax_slider), row(spacer_index_slider, index_slider), p))

# %
# ** Final check of the rendered images that will be exported as PNG files.**

# %
# Enter your preferred scaling values for vmin and vmax, based on playing with the Interactive Plot
vmin=1e-14
vmax=1e-12

# %
### Plot a sample frame with your chosen scaling values, using Helioprojective axes
figsize = (9.5, 7.5) # size of the image (resp. width and height, inches)
# Load a sample as a Sunpy NDCube instance.
# NDCube is a Sunpy object containing both the image and the metadata (including WCS data)
sample = load_ndcube_from_fits(files[0])

# %
# Creating the static plot for the sample image.
fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection":sample.wcs})
# Note that the image array is in `sample.data`.
# im = ax.imshow(sample.data, cmap=cmap_punch, norm=LogNorm(vmin=vmin, vmax=vmax))
im = ax.imshow(sample.data, cmap=cmap_punch, norm=PowerNorm(gamma=1/2.2, vmin=vmin, vmax=vmax))

lon, lat = ax.coords
lat.set_major_formatter("dd")
lon.set_major_formatter("dd")
ax.set_facecolor("black")
ax.coords.grid(color="white", alpha=.25, ls="dotted")
ax.set_xlabel("Helioprojective longitude")
ax.set_ylabel("Helioprojective latitude")
ax.set_title(f"PUNCH Level-{sample.meta.product_level} {sample.meta.product_code}"
             + f" @ {sample.meta['DATE-OBS']}"[:-4])
fig.colorbar(im, ax=ax, label="MSB")
plt.show()

# %
# Generate movie PNG frames off-screen
# ------------------------------------

# output dir
frames_dir = Path(output_dir, "frames")
figsize = (9.5, 7.5) # size of the image (resp. width and height, inches)
dpi = 100. # Resolution of the PNG file in "dots per inch"

# Create output directory for frames
os.makedirs(frames_dir, exist_ok=True)

# Preload first cube to set up once
first_cube = load_ndcube_from_fits(files[0])

# Single figure and axes created once.
fig = plt.figure(figsize=(9.5, 7.5))
ax = fig.add_subplot(111, projection=first_cube.wcs)

# Static style config applied once
ax.set_facecolor("black")
ax.coords.grid(color="white", alpha=.25, ls="dotted")
ax.set_xlabel("Helioprojective longitude")
ax.set_ylabel("Helioprojective latitude")
lon, lat = ax.coords
lat.set_major_formatter("dd")
lon.set_major_formatter("dd")

# Create image and colorbar once
im = ax.imshow(first_cube.data, cmap=cmap_punch, norm=PowerNorm(gamma=1/2.2, vmin=vmin, vmax=vmax), interpolation="nearest")
cbar = fig.colorbar(im, ax=ax, label="MSB")

# Render the first frame title
title_text = ax.set_title(f"PUNCH Level-{first_cube.meta.product_level} {first_cube.meta.product_code}"
             + f" @ {first_cube.meta['DATE-OBS']}"[:-7])

print(f"Generating {len(files)} frames...")

for i, file in enumerate(files):
    cube = load_ndcube_from_fits(file)

    # Update image data only
    im.set_data(cube.data)

    # Title update
    title_text.set_text(f"PUNCH Level-{cube.meta.product_level} {cube.meta.product_code}"
             + f" @ {cube.meta['DATE-OBS']}"[:-7])

    # png files, zero-padded numbering necessary for ingesting the filenames by ffmpeg in the right order
    frame_path = Path(frames_dir, f"frame_{i:05d}.png")

    fig.savefig(frame_path, dpi=dpi)

    # It can take long; need to show progress (here, every 10 frames).
    if (i + 1) % 10 == 0:
        print(f"  Processed {i + 1}/{len(files)} frames")

    if i+1==len(files):
      print(f"Finished processing all {len(files)} frames")

plt.close(fig)

# %
# Creating the movie with FFMPEG using the PNG frames
# --------------------------------------------------
# Your movie will be located in whatever path you define below as `movie_path`.
# A relative path will put the movie into the local folder of that Colab.
#
# If you have trouble finding your movie file on your first time through this Colab, try the /content directory if you see one in your side bar (click on the file folder) -- otherwise it should be right at the top level. It is not a persistence place, it will be deleted after your session (aka "Runtime") runs out of time or "credits".
#
# Follow the instructions in the cell above called "[Optional] Setup data persistence by mounting your personal Google Drive" to create a permanent file."""

# Full path to the output movie file
movie_path = Path(output_dir.parent.parent.parent, f"punch_movie_{stime}_to_{etime}_{product_code}.mp4")
framerate = 10   # Frames per second. That will determine movie duration.

# Crop on all sides. Cropping is done here rather than on the individual frame
# for efficiency. It is A LOT slower to tighten the frames with Matplotlib.
x_left, y_top = 40, 40 # Origin coordinates at top left corner of the crop
crop_w = 840 # width after crop
crop_h = 700 # height after crop

stream = (
    ffmpeg
    .input(os.path.join(frames_dir, "frame_%05d.png"), framerate=framerate)
    .filter('crop', crop_w, crop_h, x_left, y_top)
    # Add temporal linear interpolation for smoother video upscaled to 60 Hz
    .filter('minterpolate', fps=60, mi_mode='blend')
    .output(
        str(movie_path),
        vcodec="libx264",
        pix_fmt="yuv420p",
        crf=18
    )
    .overwrite_output()
)

# Run the stream quietly, capturing logs
out, err = stream.run(quiet=True, capture_stdout=True, capture_stderr=True)

print(f'movie created at {movie_path}')

# %
# [Optional] **Running difference images**
# ----------------------------------------
# We can create movies of running difference images with similar workflow:
#
# 1. Create the series of downsampled running difference images for the GUI
# 2. Determine your scaling parameters with the GUI
# 3. Save the series of off-screen PNG figures
# 4. Create the movie of running difference images

# %
# Series of downsampled running difference images for the GUI
# Load base image at initial time
# Load image array and downsample
base_image = load_ndcube_from_fits(files[0]).data[::binning, ::binning]
previous = base_image

cube_diff = []
diff_metadata = []
# Running difference loop: starting loop at index 1 instead of 0
for file in files[1:]:
    ndcube_obj = load_ndcube_from_fits(file)
    current = ndcube_obj.data[::binning, ::binning]
    diff_image = current - previous
    diff_image[np.isnan(diff_image)] = 0
    cube_diff.append(diff_image)
    diff_metadata.append(ndcube_obj.meta)
    previous = current

# %
# Determine your scaling parameters with the GUI
# ----------------------------------------------

# Get the low-res data for the GUI
cube = cube_diff
initial_image = cube[0]
metadata = diff_metadata

# Create the figure
p = figure(frame_width=600, frame_height=600,
           x_range=(-180, 180), y_range=(-180, 180),
           title=f"Image 0 of {len(cube)}")

# Add axis labels
p.xaxis.axis_label = "Apparent distance [Rs]"
p.yaxis.axis_label = "Apparent distance [Rs]"

vmax = 6e-13
vmin = -vmax

step_max = 1e-14
step_min = 1e-14


vmax_slider_min = 1e-14
vmax_slider_max = 1e-12
vmin_slider_min = -vmax_slider_max
vmin_slider_max = -vmax_slider_min


# Convert the Matplotlib inverted Gray colormap to a Bokeh palette
mpl_colormap = plt.get_cmap('Greys_r', 256)
bokeh_palette = [mpl_colormap(i, bytes=True) for i in range(256)]
bokeh_palette = ['#%02x%02x%02x' % (r, g, b) for r, g, b, a in bokeh_palette]
# Use a linear color mapper on the transformed data
color_mapper = LinearColorMapper(palette=bokeh_palette, low=vmin, high=vmax)

# Create a ColumnDataSource with the image data
source = ColumnDataSource(data={'image': [initial_image]})

# Display the image
p.image(image='image', x=-180, y=-180, dw=360, dh=360,
        color_mapper=color_mapper, source=source)


# Formatter that shows untransformed tick labels
colorbar_tick_formatter = CustomJSTickFormatter(code="""
    // scientific notation with 2 significant digits
    return tick.toExponential(2);
""")

# Create colorbar

color_bar = ColorBar(
    color_mapper=color_mapper,
    ticker=BasicTicker(),
    width=15,
    location=(0, 0),
    formatter=colorbar_tick_formatter
)

# Add colorbar to the figure
p.add_layout(color_bar, 'right')


# %
# Create interactive sliders
# --------------------------

# Formatter for slider tick labels
slider_tick_formatter = CustomJSTickFormatter(code="""
    return tick.toExponential(2);
""")

vmin_slider = Slider(
    start=vmin_slider_min, end=vmin_slider_max, value=vmin, step=step_min,
    title="vmin: ",
    format=slider_tick_formatter, width=350
)
vmax_slider = Slider(
    start=vmax_slider_min, end=vmax_slider_max, value=vmax, step=step_max,
    title="vmax: ", format=slider_tick_formatter, width=350
)


# JavaScript callback to update color mapper range
color_callback = CustomJS(args=dict(color_mapper=color_mapper, vmin_slider=vmin_slider, vmax_slider=vmax_slider), code="""
    color_mapper.low = vmin_slider.value;
    color_mapper.high = vmax_slider.value;
""")

vmin_slider.js_on_change('value', color_callback)
vmax_slider.js_on_change('value', color_callback)


# image index slider (time dimension)
index_slider = Slider(start=0, end=len(cube)-1, value=0, step=1, title="Image Index", width=690)

# 1) Prepare metadata for JS (one entry per image)
# Assuming you have a Python list `metadata` where each element has product_level, product_code,
# and slice.meta['DATE-OBS'] accessible as described.
js_metadata = []
for meta in metadata:
    # Extract only what you need and ensure it’s JSON-serializable (strings)
    date_obs = str(meta.get('DATE-OBS', ''))  # fallback empty string if missing
    js_metadata.append(dict(
        product_level=str(meta.product_level),
        product_code=str(meta.product_code),
        date_obs=date_obs
    ))

# Set initial title
m0 = js_metadata[0]
date0 = m0['date_obs'][:-4] if len(m0['date_obs']) >= 4 else m0['date_obs']
p.title.text = f"PUNCH Level-{m0['product_level']} {m0['product_code']} @ {date0} image 0 of {len(cube)}"

# JavaScript callback to update the image
image_callback = CustomJS(args=dict(source=source, cube=cube, p=p, n=len(cube), meta_list=js_metadata), code="""
    const data = source.data;
    const idx = cb_obj.value;
    // Update image
    data['image'] = [cube[idx]];
    const meta = meta_list[idx] || {};
    const level = meta.product_level || '';
    const code = meta.product_code || '';
    const dateObs = (meta.date_obs || '');
    // Trim last 4 chars from DATE-OBS
    const dateTrimmed = dateObs.length >= 4 ? dateObs.slice(0, -4) : dateObs;
    const prefix = `PUNCH Level-${level} ${code}`;
    p.title.text = `${prefix} @ ${dateTrimmed} — Image ${idx} of ${n}`;
    source.change.emit();
""")

index_slider.js_on_change('value', image_callback)

# Create spacer for offset
spacer_vmin_slider = Spacer(width=20)
spacer_vmax_slider = Spacer(width=100)
spacer_index_slider = Spacer(width=30)
# Display the layout
show(column(row(spacer_vmin_slider , vmin_slider, vmax_slider), row(spacer_index_slider, index_slider), p))

"""**Final check of the rendered images that will be exported as PNG files.**"""

# Scaling values. Using percentiles on intensity, instead of actual intensity values
vmax = 2e-13
vmin = -vmax
print(f"vmin = {vmin:0.1e}, vmax = {vmax:0.1e}")


# Load initial data at full resolution to get a wcs-consistent coordinate system
init_wcs = load_ndcube_from_fits(files[1]).wcs
init_image_hr = load_ndcube_from_fits(files[0]).data
init_diff_hr = load_ndcube_from_fits(files[1]).data - init_image_hr

# Plot a sample
fig, ax = plt.subplots(figsize=(9.5, 7.5), subplot_kw={"projection":init_wcs})

im = ax.imshow(init_diff_hr, vmin=vmin, vmax=vmax, cmap='Greys_r')

lon, lat = ax.coords
lat.set_major_formatter("dd")
lon.set_major_formatter("dd")
ax.set_facecolor("black")
ax.coords.grid(color="white", alpha=.25, ls="dotted")
ax.set_xlabel("Helioprojective longitude")
ax.set_ylabel("Helioprojective latitude")
ax.set_title(f"Running Difference - PUNCH Level-{diff_metadata[0].product_level} {diff_metadata[0].product_code}"
             + f" @ {diff_metadata[0]['DATE-OBS']}"[:-4])
fig.colorbar(im, ax=ax, label="MSB Difference")
plt.show()

# %
# Save the series of off-screen PNG figures
# -----------------------------------------

# Output directory for the difference images
frames_diff_dir = Path(output_dir, "frames")
# Create output directory for frames
os.makedirs(frames_diff_dir, exist_ok=True)

# Plot initial sample
fig, ax = plt.subplots(figsize=(9.5, 7.5), subplot_kw={"projection":init_wcs})
# Static style config applied once
ax.set_facecolor("black")
ax.coords.grid(color="white", alpha=.25, ls="dotted")
ax.set_xlabel("Helioprojective longitude")
ax.set_ylabel("Helioprojective latitude")
lon, lat = ax.coords
lat.set_major_formatter("dd")
lon.set_major_formatter("dd")

# Create image and colorbar once
im = ax.imshow(init_diff_hr, cmap='Greys_r', vmin=vmin, vmax=vmax)
cbar = fig.colorbar(im, ax=ax, label="MSB difference")

# Render the first frame title
title_text = ax.set_title(f"Running Difference - PUNCH Level-{diff_metadata[0].product_level} {diff_metadata[0].product_code}"
             + f" @ {diff_metadata[0]['DATE-OBS']}"[:-4])

print(f"Generating {len(files)} frames...")

# Load base image at initial time and full resolution
base_image = load_ndcube_from_fits(files[0]).data
previous = base_image

for i, file in enumerate(files[1:]):

    ndcube_obj = load_ndcube_from_fits(file)
    meta = ndcube_obj.meta
    current = ndcube_obj.data
    diff_image = current - previous
    diff_image[np.isnan(diff_image)] = 0
    previous = current

    # Update image data only
    im.set_data(diff_image)

    # Title update
    title_text.set_text(f"PUNCH Level-{meta.product_level} {meta.product_code}"
             + f" @ {meta['DATE-OBS']}"[:-4])

    # png files, zero-padded numbering necessary for ingesting the filenames by ffmpeg in the right order
    frame_path = Path(frames_diff_dir, f"frame_diff_{i:05d}.png")

    fig.savefig(frame_path, dpi=dpi)

    # It can take long; need to show progress (here, every 10 frames).
    if (i + 1) % 10 == 0:
        print(f"  Processed {i + 1}/{len(files)} frames")

    if i+1==len(files[1:]):
      print(f"Finished processing all {len(files)} frames")

plt.close(fig)

# Full path to the output movie file
movie_path = Path(output_dir.parent.parent.parent, f"punch_diff_movie_{stime}_to_{etime}_{product_code}.mp4")
framerate = 10   # Frames per second. That will determine movie duration.

# Crop on all sides. Cropping is done here rather than on the individual frame
# for efficiency. It is A LOT slower to tighten the frames with Matplotlib.
x_left, y_top = 40, 40 # Origin coordinates at top left corner of the crop
crop_w = 840 # width after crop
crop_h = 700 # height after crop

stream = (
    ffmpeg
    .input(os.path.join(frames_diff_dir, "frame_diff_%05d.png"), framerate=framerate)
    .filter('crop', crop_w, crop_h, x_left, y_top)
    # Add temporal linear interpolation for smoother video upscaled to 60 Hz
    .filter('minterpolate', fps=60, mi_mode='blend')
    .output(
        str(movie_path),
        vcodec="libx264",
        pix_fmt="yuv420p",
        crf=18
    )
    .overwrite_output()
)

# Run the stream quietly, capturing logs
out, err = stream.run(quiet=True, capture_stdout=True, capture_stderr=True)

print(f'movie created at {movie_path}')
