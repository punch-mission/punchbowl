Despike
=======

Cosmic rays passing through an instrument CCD can leave behind bright spots or streaks in the collected image, depending on the angle of incidence.
This despiking module aims to identify and remove cosmic ray artifacts from a sequence of images.

Concept
-------

A PUNCH roll sequence consists of three polarized exposures, a clear exposure, and three more polarized exposures.
As cosmic ray artifacts are temporally transient, a given cluster of pixels will likely only show an artifact in one of these images.
N input images from the roll sequence (three minimum) are high-pass-filtered, and the median is computed for each pixel from the resulting N-1 lowest values per pixel.
From here z-score filtering is used to identify outlier pixels from cosmic ray strikes, which are removed and filled in from surrounding valid pixels.

Applying correction
-------------------

The correction is carried out primarily using the ``punchbowl.level1.despike.despike_polseq`` function:

.. autofunction:: punchbowl.level1.despike.despike_polseq
    :no-index:

If you wish to incorporate this as a Prefect task in a custom pipeline,
using something like the ``punchbowl.level1.despike.despike_polseq_task`` is recommended.
