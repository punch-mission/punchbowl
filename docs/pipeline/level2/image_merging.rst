Image Merging
=============

Once spacecraft data has been projected into a unified frame and further image processing has been completed, these data can be merged into one virtual observation.

Concept
-------

With each observation in an identical frame, image merging can be completed by a weighted average based on the input data uncertainty. Where overlap exists between instruments, a weighted average favors data with lower uncertainty.

Applying correction
-------------------

Image merging is carried out in the ``punchbowl.level2.merge.merge_many_polarized_task`` function for polarized data input and the ``punchbowl.level2.merge.merge_many_clear_task`` function for clear (unpolarized) data input:

.. autofunction:: punchbowl.level2.merge.merge_many_polarized_task
    :no-index:

.. autofunction:: punchbowl.level2.merge.merge_many_clear_task
    :no-index:

If you wish to incorporate this as a Prefect task in a custom pipeline,
using something like the ``punchbowl.level2.merge.merge_many_polarized_task`` or ``punchbowl.level2.merge.merge_many_clear_task`` is recommended.
