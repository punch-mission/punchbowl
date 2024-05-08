Quartic Fit Correction
=======================

Concept
---------

The quartic fit correction takes care of the flat field, gain, and other pixel-by-pixel effects.
There are five coefficients (a, b, c, d, e) for each pixel that define a function for correction:

.. math::

    X_{i,j} = a_{i,j}+b_{i,j}*DN_{i,j}+c_{i,j}*DN_{i,j}^2+d_{i,j}*DN_{i,j}^3+e_{i,j}*DN_{i,j}^4

For pixel at coordinate (i, j), the original image values in DN are corrected to be X using the five coefficients
in a polynomial correction.

Applying correction
---------------------

The correction is carried out primarily in the ``punchbowl.level1.quartic_fit.photometric_calibration`` function:

.. autofunction:: punchbowl.level1.quartic_fit.photometric_calibration
    :no-index:

If you wish to incorporate this as a Prefect task in a custom pipeline,
using something like the ``punchbowl.level1.quartic_fit.perform_quartic_fit_task`` is recommended.

Deriving coefficients
----------------------
