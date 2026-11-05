Polarization Resolution
=======================

Data enters level 3 processing in the MZP (minus 60 degrees, zero, and plus 60 degrees) representation of polarized light with respect to solar north in the unified mosaic frame. These observations can be transformed into brightness (tB) and polarized brightness (pB).

Concept
-------

The polarization correction is primarily carried out by the `solpolpy <https://github.com/punch-mission/solpolpy>`_ framework, with a module within the punchbowl calling this code. See this code for full documentation.

When MZP standard deviations are supplied, the conversion propagates their variances
using the same per-pixel angle map as the signal. For independent MZP errors,

.. math::

   \sigma_{tB}^2 &= \frac{4}{9}\sum_i \sigma_i^2, \\
   \sigma_{pB}^2 &= \frac{16}{9}\sum_i \cos^2[2(\theta_i-\alpha)]\sigma_i^2, \\
   \sigma_{pB'}^2 &= \frac{16}{9}\sum_i \sin^2[2(\theta_i-\alpha)]\sigma_i^2.

Here the polarizer angles are -60, 0 and +60 degrees, and alpha is supplied by
solpolpy. This assumes fixed geometry and independent input errors; cross-channel
covariance and shared background or calibration errors are not included. A resolved
triplet is unsupported if any contributing MZP value or error is invalid or masked.

Applying correction
-------------------

Polarization resolution is carried out in the ``punchbowl.level3.polarization.convert_polarization`` function:

.. autofunction:: punchbowl.level3.polarization.convert_polarization
    :no-index:

If you wish to incorporate this as a Prefect task in a custom pipeline,
using something like the ``punchbowl.level3.polarization.convert_polarization`` is recommended.
