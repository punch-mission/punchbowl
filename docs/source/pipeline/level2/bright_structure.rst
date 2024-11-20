Bright Structure Identification
===============================

Bright structure identification is a necessary step for clearing the data of erroneous signal from bright structures visible in the instrument field of view, such as terrestrial aurora.

Concept
-------

Given a time sequence of images, this module identifies "spikes" that exceed a threshold in a single frame. The details of this process are included in the documented function below.

Applying correction
-------------------

Bright structure identification and marking is carried out in the ``punchbowl.level2.bright_structure.run_zspike`` function:

.. autofunction:: punchbowl.level2.bright_structure.run_zspike
    :no-index:

If you wish to incorporate this as a Prefect task in a custom pipeline,
using something like the ``punchbowl.level2.bright_structure.identify_bright_structures_task`` is recommended.
