PSF Correction
=======================

The point spread function (PSF) of the imagers varies across the field-of-view (FOV) of each imager.
The PSFs have to be homogenized to be similar before creating mosaic images. Otherwise, the tails of the PSFs would
combine in messy ways and create artifacts. In addition, the `pointing refinement algorithm <pointing.rst>`_ expects a
uniform PSF across the FOV.

Concept
---------

The PSF correction is carried out by the `regularizepsf package <https://github.com/punch-mission/regularizepsf>`_.
The description of the algorithm and code
is `available here <https://punch-mission.github.io/regularizepsf/concepts.html>`_.
There's also an
`accompanying paper with more rigorous description <https://punch-mission.github.io/regularizepsf/concepts.html>`_.


Applying correction
---------------------

The correction is carried out primarily in the ``punchbowl.level1.psf.correct_psf_task`` function:

.. autofunction:: punchbowl.level1.psf.correct_psf_task
    :no-index:

If you wish to incorporate this as a Prefect task in a custom pipeline,
using something like the ``punchbowl.level1.psf.correct_psf_task`` is recommended.
