Pointing Correction
===================

While individual spacecraft align themselves on orbit using star tracker observations, a refined
correction using stellar positions must be applied to allow for precise data reprojection and
merging later in the pipeline.

Concept
-------

The pointing correction is carried out by the `thuban <https://github.com/punch-mission/thuban>`_ package, with a module within the punchbowl calling this code.

Applying correction
-------------------

The correction is carried out primarily in the ``punchbowl.level1.alignment.align_task`` function:

.. autofunction:: punchbowl.level1.alignment.align_task
    :no-index:

If you wish to incorporate this as a Prefect task in a custom pipeline,
using something like the ``punchbowl.level1.alignment.align_task`` is recommended.
