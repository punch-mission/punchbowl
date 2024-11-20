Image Resampling
=================

To later merge each individual PUNCH spacecraft observation into a unified virtual observatory later in the pipeline, first each observation must be reprojected into a standardized frame.

Concept
-------

Each observation contains a corresponding World Coordinate System (WCS), describing the physical coordinates of each pixel in the data. With an output WCS defined for the full PUNCH field of view observation, the AstroPy `reproject <https://github.com/astropy/reproject>`_ package can be used to transform the data from the spacecraft frame into the full mosaic frame.

Note that here we use the `adaptive reprojection <https://reproject.readthedocs.io/en/stable/celestial.html#adaptive-resampling>`_ methodology based on a `DeForest (2004) <https://link.springer.com/article/10.1023/B:SOLA.0000021743.24248.b0>`_ algorithm - better preserving structure and photometry.

Applying correction
-------------------

Image resampling is carried out in the ``punchbowl.level2.resmaple.reproject_cube`` function:

.. autofunction:: punchbowl.level2.resample.reproject_cube
    :no-index:

If you wish to incorporate this as a Prefect task in a custom pipeline,
using something like the ``punchbowl.level2.resmaple.reproject_many_flow`` is recommended.
