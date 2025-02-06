Introduction
=============

What is PUNCH?
--------------
PUNCH is a NASA Small Explorer (SMEX) mission to better understand how the mass and energy of
the Sun’s corona become the solar wind that fills the solar system.
Four suitcase-sized satellites will work together to produce images of the entire inner solar system around the clock.
You can learn more at the `PUNCH website <https://punch.space.swri.edu/>`_.

Where does `punchbowl` fit in?
--------------------------------------
``punchbowl`` is the data reduction pipeline code for the PUNCH mission. The pipeline, as shown below,
consists of several segments of processing.

.. figure:: ./images/PUNCH_data_flow.png
    :alt: PUNCH pipeline schematic
    :width: 800px
    :align: center

    A schematic overview of the PUNCH data processing pipeline


These segments are the following:

- Raw to *Level 0*: converts raw satellite data to FITS images
- Level 0 to *Level 1*: basic image calibration
- Level 1 to *Level 2*: polarization resolution, image merging, quality marking
- Level 2 to *Level 3*: background subtraction

We identify these segments by their finishing level, i.e. the Level 1 products come from the Level 0 to Level 1 segment
which can be called just the Level 1 segment for short. The processing description and code you'll find here is
organized in this manner.

PUNCH and Python
----------------

The PUNCH framework is built using Python - an object-oriented language with a large user / code base in astronomy and solar physics. The pipeline and tools for querying / loading PUNCH data use the Python language, along with the SunPy and Astropy software libraries. A number of useful tutorials exist online, including the official `Python tutorial <https://docs.python.org/3/tutorial/index.html>`_ and the `Hitchhiker's Guide to Python <https://docs.python-guide.org>`_.

In addition to scripts and modules, Python notebooks provide a great way to execute and document a sequence of code cells, with visualizations directly in-line. It's a sort of analogue of the classic research notebook. The `SunPy example gallery <https://docs.sunpy.org/en/stable/generated/gallery/index.html>`_ provides a great series of example notebooks, which are an additional great tool for learning Python.
