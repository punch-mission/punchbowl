Despike
=======

The goal of despiking is to remove cosmic ray hits from images.

Concept
-------

Despiking is carried out by our implementation
of `Astropy's Astro-SCRAPPY algorithm <https://github.com/astropy/astroscrappy>`_.
The algorithm works on a single image at a time and is based on Pieter van Dokkum's L.A.Cosmic algorithm.
It identifies cosmic rays in images by exploiting their sharp, high-contrast edges using a Laplacian
edge detection algorithm. This approach distinguishes cosmic rays from real astronomical objects by their
characteristic sharpness and lack of spatial coherence. The technique enhances the Laplacian-filtered image,
isolates significant outliers, and then removes or replaces the cosmic ray pixels to preserve image integrity.

Applying correction
-------------------

The correction is carried out primarily using the ``astroscrappy.detect_cosmics`` function:

.. autofunction:: astroscrappy.detect_cosmics
    :no-index:

If you wish to incorporate this as a Prefect task in a custom pipeline,
using something like the ``punchbowl.level1.despike.despike_task`` is recommended.
