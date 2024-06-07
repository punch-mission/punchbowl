Initial Uncertainty Estimation
===============================

We propagate uncertainty through the pipeline. This module initializes the
uncertainty based on the bias level, the dark level, the gain, read noise level,
bitrate, and signal levels.

Concept
---------

The initial uncertainty is characterized in level 1 data products by passing the input data file and associated
metadata into a subroutine which computes and inserts these uncertainty values in place. Using CCD / instrument
characteristics such as the bias level, the dark level, the gain, the read noise level, and the signal bitrate,
an estimation of the noise levels can be computed.

Photon / shot noise is generated from the gain and input signal.
The dark noise is generated from the dark level and a poisson distribution. The read noise is generated from a normal
distribution using the provided read noise level. These individual noise terms are added in quadrature, and returned
from a subroutine. The initial uncertainty estimate is then folded into the data object as a ratio of the noise level
with the provided / computed data photon counts.

Applying correction
---------------------

The correction is carried out primarily in the ``punchbowl.level1.initial_uncertainty.compute_uncertainty`` function:

.. autofunction:: punchbowl.level1.initial_uncertainty.compute_uncertainty
    :no-index:

If you wish to incorporate this as a Prefect task in a custom pipeline,
using something like the ``punchbowl.level1.initial_uncertainty.update_initial_uncertainty_task`` is recommended.
