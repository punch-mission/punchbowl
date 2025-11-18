Despike
=======

The goal of despiking is to remove cosmic ray hits from images.

Concept
-------

Despiking hinges on looking for pixels that have high variability in a single roll state. 
The idea is that a cosmic ray will hit only one image. 
While the polarization changes between images, the satellite is stable. 
Thus up to the polarization differences, the scene is largely the same. A cosmic ray will look 
anomalous if compared to the average frame in a roll state. 

Applying correction
-------------------

The correction is carried out primarily using the ``punchbowl.level1.despike.despike_polseq`` function:

.. autofunction:: punchbowl.level1.despike.despike_polseq
    :no-index:

If you wish to incorporate this as a Prefect task in a custom pipeline,
using something like the ``punchbowl.level1.despike.despike_polseq_task`` is recommended.
