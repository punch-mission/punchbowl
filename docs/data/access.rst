Accessing PUNCH Data
====================

Which Product to look for?
--------------------------
Recommended Level 3 Products
-----------------------------

For most science use cases, the recommended starting point is the Level 3 **CAM**
(Clear/unpolarized low-noise science mosaic) and **PAM** (Polarized low-noise science mosaic) products.
Both are fully background and starfield subtracted and mosaicked across the PUNCH field of view,
giving you the cleanest possible signal for heliospheric science.
CAM provides total brightness (B) with uncertainty, while PAM additionally resolves
polarized brightness (pB) and its radial component (pB'), making PAM the product of
choice for studies of the CMEs, shocks, solar wind etc. These products
represent one full spacecraft rotation cycle with a significant
improvement in signal-to-noise relative to other products.

Researchers who need intermediate data products — for example to inspect F-corona
subtraction before the final background removal, or to carry out their own starfield
subtraction — should consider the **CIM** (Clear/unpolarized science trefoil with F-corona subtraction)
and **PIM** (Polarized science trefoil with F-corona subtraction) products. CIM and PIM
retain the trefoil geometry of the full PUNCH field of view but stop short of the full
background subtraction pipeline applied to CAM and PAM. They are particularly useful for
validating custom reduction steps or for science cases that are sensitive to how the
background model is constructed.

If you are unsure which product to start with, download a CAM or PAM file first and work backward to CIM/PIM only if your analysis requires it.

Downloading Data
----------------
Data output from the PUNCH data processing pipeline are stored and accessible through the Solar Data Analysis Center (SDAC)
- a portal for hosting through tools such as the Virtual Solar Observatory (VSO).
From here PUNCH data products can be queried and requested for download using metadata within the data products.
See `this example <https://punchbowl.readthedocs.io/en/latest/auto_examples/querying_data.html#sphx-glr-auto-examples-querying-data-py>`_ on how to query the VSO using SunPy's Fido API.

If that example is not working properly, you can also pull data directly from the SDAC using ``wget``.

.. code-block:: bash

    wget -r -l1 --no-parent --no-directories -A "PUNCH_L2_CTM_20250921*_v0h.fits" -R "*.html*,index*,*tmp*" https://umbra.nascom.nasa.gov/punch/2/CTM/2025/09/21/

The above example would pull data for the L2_CTM products on 2025-09-21.
Change the path and date according to what product you wish to download.

PUNCH data will also be accessible soon using the Helioviewer tool,
where it can be quickly visualized and stitched together with other observations for context.

Reading Data
------------
Standard PUNCH data is stored as a standards-compliant FITS file, which bundles the primary data along with secondary data and metadata fully describing the observation.
Each file is named with a convention that uniquely identifies the product - a sample being 'PUNCH_L3_PAM_20230704000000_v1.fits' - where L3 defines the data level,
PAM is an example of a particular data product code, 20230704000000 is a timestamp in the format yyyymmddhhmmss, and _v1 is the version of the data.

For most end-users the primary data of interest are PAM (low-noise full frame data gathered over one full spacecraft rotation cycle) and PTM (high-cadence trefoil mosaics).

PUNCH FITS files are RICE compressed, reducing the overall file size while preserving data fidelity.
Due to this compression, the zeroth HDU of each output data file contains information about the compression scheme.
The first HDU (``hdul[1]``) contains the primary data array, along with an astropy header string describing that data.
The second HDU (``hdul[2]``) contains the uncertainty array - corresponding on a pixel-by-pixel basis with the primary data array.

These data are compatible with standard astropy FITS libraries, and can be read in as following the example,

.. code-block:: python

    filename = 'example_data/PUNCH_L3_PAM_20240620000000.fits'

    with fits.open(filename) as hdul:
        data = hdul[1].data
        header = hdul[1].header
        uncertainty = hdul[2].data

These data can also be bundled together as an NDCube object, either manually or using some of the bundled IO tools within punchbowl. For instance,

.. code-block:: python

    from punchbowl.data.punch_io import load_ndcube_from_fits

    filename = 'example_data/PUNCH_L3_PAM_20240620000000.fits'

    cube = load_ndcube_from_fits(filename)

See `this code example <https://punchbowl.readthedocs.io/en/latest/auto_examples/data_guide.html#sphx-glr-auto-examples-data-guide-py>`_ on how to use data.

Data Projections
----------------
The PUNCH WFI instruments extend their field of view out to around 45-degrees from the Sun,
creating a meshed virtual observatory extending to a diameter of nearly 180 solar radii.
The wide nature of this field of view requires attention to the data projection being used for these data.

For NFI data, the standard projection is a Gnomonic (TAN) coordinate system with distortion,
a standard system employed for many data peering closer towards the sun.

For individual WFI data, an azimuthal perspective (AZP) coordinate system with distortion is used.

For full mosaics that combine data from WFI and NFI, an azimuthal equidistant (ARC) coordinate system is used,
with data from each spacecraft frame aligned and projected to this standardized frame.

Each data contains a set of World Coordinate System (WCS) parameters that describe the coordinates of the data,
in both a helioprojective and celestial frame.
