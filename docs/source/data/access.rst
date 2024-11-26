Accessing PUNCH Data
====================

Downloading Data
----------------
Data output from the PUNCH data processing pipeline will be stored and accessible through the Solar Data Analysis Center (SDAC) - a portal for hosting through tools such as the Virtual Solar Observatory (VSO). From here PUNCH data products can be queried and requested for download using metadata within the data products.

PUNCH data will also be accessible using the helioviewer tool, where it can be quickly visualized and stitched together with other observations for context.

Reading Data
------------
Standard PUNCH data is stored as a standards-compliant FITS file, which bundles the primary data along with secondary data and metadata fully describing the observation. Each file is named with a convention that uniquely identifies the product - a sample being 'PUNCH_L3_PAM_20230704000000_v1.fits' - where L3 defines the data level, PAM is an example of a particular data product code, 20230704000000 is a timestamp in the format yyyyhhmmhhmmss, and _v1 is the version of the data (used in reprocessing).

For most end-users the primary data of interest are PAM (low-noise full frame data gathered over one full spacecraft rotation cycle) and PTM (high-cadence trefoil mosaics).

PUNCH FITS files are RICE compressed, reducing the overall file size while preserving data fidelity. Due to this compression, the zeroth HDU of each output data file contains information about the compression scheme. The first HDU (hdul[1]) contains the primary data array, along with an astropy header string describing that data. The second HDU (hdul[2]) contains the uncertainty array - corresponding on a pixel-by-pixel basis with the primary data array.

These data are compatible with standard astropy FITS libraries, and can be read in as following the example,

.. code-block:: python

    filename = 'example_data/PUNCH_L3_PAM_20240620000000.fits'

    with fits.open(filename) as hdul:
        data = hdul[1].data
        header = hdul[1].header
        uncertainty = hdul[2].data

These data can also be bundled together as an NDCube object, either manually or using some of the bundled IO tools within punchbowl.

Data Projections
----------------
The PUNCH WFI instruments extend their field of view out to around 45-degrees from the Sun, creating a meshed virtual observatory extending to a diameter of nearly 180 solar radii. The wide nature of this field of view requires attention to the data projection being used for these data.

For NFI data, the standard projection is a Gnomonic (TAN) coordinate system with distortion, a standard system employed for many data peering closer towards the sun.

For individual WFI data, an azimuthal perspective (AZP) coordinate system with distortion is used.

For full mosaics that combine data from WFI and NFI, an azimuthal equidistant (ARC) coordinate system is used, with data from each spacecraft frame aligned and projected to this standardized frame.

Each data contains a set of World Coordinate System (WCS) parameters that describe the coordinates of the data, in both a helioprojective and celestial frame.
