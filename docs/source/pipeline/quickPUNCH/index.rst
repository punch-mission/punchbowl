QuickPUNCH
==========

QuickPUNCH data follows a similar processing pipeline to that of level 2 PUNCH data, with some pipeline differences to account for faster processing latency. A NFI low-latency QuickPUNCH data product spans the same field of view as standard NFI data, with a mosaic low-latency QuickPUNCH data product extending out to a reduced field of view, to reduce gaps between trefoils. Both data products are reduced to 1024 by 1024 pixels. A F-corona model is generated for these data products from the preceding month of data, and are subtracted and stored as QuickPUNCH calibration files.

A notebook below provides an example of working with these data.

.. toctree::
    quickpunch_data_guide.ipynb
