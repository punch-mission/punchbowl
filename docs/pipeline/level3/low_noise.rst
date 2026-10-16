Low Noise Data Generation
=========================

From individual trefoil mosaic observations, low-noise data can be collected and summed over one full synchronized spacecraft rotation cycle.

Concept
-------

A set of higher cadence trefoil mosaics are collected, read, and assembled into a low-noise mosaic. Where overlapping data exists on a pixel-by-pixel basis, an average of all available observations is taken, propagating the associated uncertainties as well. This data spans the full field of view without the typical gaps in spatial coverage, and with lower noise characteristics.

Only Level 3 PTM inputs on the PIM-to-PAM path use the corrected merger.
These PTMs contain tB, pB and pBp after starfield subtraction and polarization
conversion. Level 2 PTMs contain MZP and retain the original merging behavior.
For independent observations, overlapping samples are combined with inverse-variance
weights:

.. math::

   w_k = \sigma_k^{-2}, \qquad
   \bar I = \frac{\sum_k w_k I_k}{\sum_k w_k}, \qquad
   \sigma_{\bar I} = \sqrt{\frac{1}{\sum_k w_k}}.

The returned uncertainty describes the weighted mean, rather than the RMS of input
errors. Finite zero and negative signals remain valid. Masked samples, nonfinite
signals, and nonpositive or nonfinite errors contribute no weight. Pixels without
support have zero data and infinite uncertainty. Errors shared between observations,
such as a common background model, are not included in this independent-error formula.

PAM requires propagated input standard deviations. CAM, QuickPUNCH, and all Level 2
products retain the original shared merger and uncertainty calculation.

Applying correction
-------------------

Image resampling is carried out in the ``punchbowl.level3.low_noise.create_low_noise_task`` function:

.. autofunction:: punchbowl.level3.low_noise.create_low_noise_task
    :no-index:

If you wish to incorporate this as a Prefect task in a custom pipeline,
using something like the ``punchbowl.level3.low_noise.create_low_noise_task`` is recommended.
