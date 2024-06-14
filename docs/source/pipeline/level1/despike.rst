Despike
==========

The goal of despiking is to remove cosmic ray hits from images.

Concept
---------

Despiking is carried out by our implementation
of `DeForest's spikejones algorithm <https://github.com/drzowie/solarpdl-tools/blob/79d431d937bab6178eb68bec229eee59614233b3/image/spikejones.pdl#L12>`_.
The algorithm works on a single image at a time. It applies an unsharp mask
to an image and compares that to a smoothed copy. Pixels with a significant difference
are considered spikes.

Applying correction
---------------------

The correction is carried out primarily in the ``punchbowl.level1.despike.spikejones`` function:

.. autofunction:: punchbowl.level1.despike.spikejones
    :no-index:

If you wish to incorporate this as a Prefect task in a custom pipeline,
using something like the ``punchbowl.level1.despike.despike_task`` is recommended.
