Low Noise Data Generation
=========================

From individual trefoil mosaic observations, low-noise data can be collected and summed over one full synchronized spacecraft rotation cycle.

Concept
-------

A set of higher cadence trefoil mosaics are collected, read, and assembled into a low-noise mosaic. Where overlapping data exists on a pixel-by-pixel basis, an average of all available observations is taken, propagating the associated uncertainties as well. This data spans the full field of view without the typical gaps in spatial coverage, and with lower noise characteristics.

Applying correction
-------------------

Image resampling is carried out in the ``punchbowl.level3.low_noise.create_low_noise_task`` function:

.. autofunction:: punchbowl.level3.low_noise.create_low_noise_task
    :no-index:

If you wish to incorporate this as a Prefect task in a custom pipeline,
using something like the ``punchbowl.level3.low_noise.create_low_noise_task`` is recommended.
