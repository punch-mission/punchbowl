Pointing Correction
===================

While individual spacecraft align themselves on orbit using star tracker observations, a refined
correction using stellar positions must be applied to allow for precise data reprojection and
merging later in the pipeline.

Concept
-------

The pointing correction begins by using Astrometry.net as a lost-in-space pointing solver.
This means we do not have to rely on spacecraft pointing being reliable making the algorithm more
robust to South Atlantic Anomaly events.
After the rough solving by Astrometry.net we refine the pointing using our known distortion model.
We select many asterisms of stars and simultaneously adjust the pointing parameters
(CROTA, CRVAL1, CRVAL2, platescale, and PV values) to match the catalog
expected position to the extracted position of stars in the image.
The algorithms execute many of these fitting steps and then take the median of each parameter as the average
world coordinate system to adopt.

Applying correction
-------------------

The correction is carried out primarily in the ``punchbowl.level1.alignment.align_task`` function:

.. autofunction:: punchbowl.level1.alignment.align_task
    :no-index:

If you wish to incorporate this as a Prefect task in a custom pipeline,
using something like the ``punchbowl.level1.alignment.align_task`` is recommended.

Deriving distortion correction
------------------------------

The distortion model is determined from PSF-corrected images.
The pointing is first solved and then any remaining error is
measured as the distortion offset. This process iterates back and forth
between pointing and distortion solving to account for their interconnected error.
