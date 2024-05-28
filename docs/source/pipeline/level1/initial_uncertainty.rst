Initial Uncertainty Estimation
===============================

We propagate uncertainty through the pipeline. This module initializes the
uncertainty based on the bias level, the dark level, the gain, read noise level,
bitrate, and signal levels.

Concept
---------

TODO


Applying correction
---------------------

The correction is carried out primarily in the ``punchbowl.level1.initial_uncertainty.compute_uncertainty`` function:

.. autofunction:: punchbowl.level1.initial_uncertainty.compute_uncertainty
    :no-index:

If you wish to incorporate this as a Prefect task in a custom pipeline,
using something like the ``punchbowl.level1.initial_uncertainty.update_initial_uncertainty_task`` is recommended.
