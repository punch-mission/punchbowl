F-Corona Removal
================

The F-corona - light emitted from scatted dust within the solar system - must be removed from data to uncover the desired usable signal.

Concept
-------

For any individual data a corresponding F-corona model calibration file is retrieved and subtracted from the data, leaving behind the usable signal.

Applying correction
--------------------

F-corona removal is carried out in the ``punchbowl.level3.f_corona_model.subtract_f_corona_background`` function:

.. autofunction:: punchbowl.level3.f_corona_model.subtract_f_corona_background
    :no-index:

If you wish to incorporate this as a Prefect task in a custom pipeline,
using something like the ``punchbowl.level3.f_corona_model.subtract_f_corona_background_task`` is recommended.

Deriving F-corona model
-----------------------

The F-corona signal varies in time, however, this can be calculated using a span of data to provide a representative distribution for that span of time. By gathering spatially co-aligned observations, a quadratic programming technique can be used to extract the baseline emission from F-corona in each pixel, dusting away faster variations.
