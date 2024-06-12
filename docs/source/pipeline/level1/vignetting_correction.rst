Vignetting Correction
=======================

Concept
---------

Optical vignetting describes the drop in brightness towards the edge of images captured through an optical path.
The vignetting correction module applies measured calibration maps to undo the effects of optical vignetting.
Vignetting calibration maps are spacecraft and polarizer-state dependent,
and are measured pre-flight and updated using starfield measurements during commissioning
and during the life of the mission.

For an input data array (I) and corresponding vignetting correction array (VG),
the corrected data array (I') for each corresponding pixel i,j is computed via:

.. math::

    I'_{i,j} = I_{i,j} /  VG_{i,j}

Applying correction
---------------------

The correction is carried out primarily in the ``punchbowl.level1.vignette.correct_vignetting_task`` function:

.. autofunction:: punchbowl.level1.vignette.correct_vignetting_task
    :no-index:

If you wish to incorporate this as a Prefect task in a custom pipeline,
using something like the ``punchbowl.level1.vignette.correct_vignetting_task`` is recommended.

Deriving correction
----------------------

Vignetting correction calibration files are derived pre-flight from measurements,
and will be updated during commissioning using starfield brightness data, via a separate module.
