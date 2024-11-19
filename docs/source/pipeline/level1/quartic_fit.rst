Quartic Fit Correction
======================

Concept
-------

The quartic fit correction takes care of the flat field, gain, and other pixel-by-pixel effects.
There are five coefficients (a, b, c, d, e) for each pixel that define a function for correction:

.. math::

    X_{i,j} = a_{i,j}+b_{i,j}*DN_{i,j}+c_{i,j}*DN_{i,j}^2+d_{i,j}*DN_{i,j}^3+e_{i,j}*DN_{i,j}^4

For pixel at coordinate (i, j), the original image values in DN are corrected to be X using the five coefficients
in a polynomial correction.

Vignetting correction
---------------------

Vignetting is folded into the quartic fit correction. Optical vignetting describes the drop in brightness towards the edge of images captured through an optical path.
The vignetting correction module applies measured calibration maps to undo the effects of optical vignetting.
Vignetting calibration maps are spacecraft and polarizer-state dependent,
and are measured pre-flight and updated using starfield measurements during commissioning
and during the life of the mission.

For an input data array (I) and corresponding vignetting correction array (VG),
the corrected data array (I') for each corresponding pixel i,j is computed via:

.. math::

    I'_{i,j} = I_{i,j} /  VG_{i,j}

Applying correction
-------------------

The correction is carried out primarily in the ``punchbowl.level1.quartic_fit.photometric_calibration`` function:

.. autofunction:: punchbowl.level1.quartic_fit.photometric_calibration
    :no-index:

If you wish to incorporate this as a Prefect task in a custom pipeline,
using something like the ``punchbowl.level1.quartic_fit.perform_quartic_fit_task`` is recommended.

Deriving correction
-------------------

Quartic fit / vignetting correction calibration files are derived pre-flight from measurements,
and will be updated during commissioning using starfield brightness data, via a separate module.
