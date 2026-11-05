Low Noise Data Generation
=========================

From individual trefoil mosaic observations, low-noise data can be collected and summed over one full synchronized spacecraft rotation cycle.

Concept
-------

A set of higher cadence trefoil mosaics are collected, read, and assembled into a low-noise mosaic. Where overlapping data exists on a pixel-by-pixel basis, an average of all available observations is taken, propagating the associated uncertainties as well. This data spans the full field of view without the typical gaps in spatial coverage, and with lower noise characteristics.

Clear and polarized low-noise products use the shared Level 2 merger. Propagated
tB, pB and pBp standard deviations determine the inverse-variance weights for PAM:

.. math::

   w_k = \sigma_k^{-2}, \qquad
   \bar I = \frac{\sum_k w_k I_k}{\sum_k w_k}.

For consistency with clear products, the output uncertainty retains the existing
RMS convention over valid contributing samples:

.. math::

   \sigma_{\mathrm{out}} = \sqrt{\frac{1}{N}\sum_k \sigma_k^2}.

This is the RMS of the input standard deviations, not the statistical uncertainty
of the weighted mean. Finite zero and negative signals remain valid. Masked samples,
nonfinite signals, and nonpositive or nonfinite errors contribute neither weight nor
RMS uncertainty. Pixels without support have zero data, infinite uncertainty and
an output mask. Shared background and calibration covariance are not estimated.

Applying correction
-------------------

Image resampling is carried out in the ``punchbowl.level3.low_noise.create_low_noise_task`` function:

.. autofunction:: punchbowl.level3.low_noise.create_low_noise_task
    :no-index:

If you wish to incorporate this as a Prefect task in a custom pipeline,
using something like the ``punchbowl.level3.low_noise.create_low_noise_task`` is recommended.
