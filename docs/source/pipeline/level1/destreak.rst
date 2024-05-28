Destreaking
=======================

De-streaking accounts for smearing effects from the PUNCH shutterless frame transfer mode of
operation. Each detector continues exposing during readout as the science image is moved off of
the active detector area and onto the storage area.

Concept
---------

First, we construct a matrix that contains the streaking operation. It is inverted and returned
by ``punchbowl.level1.destreak.streak_correction_matrix``. This is multiplied into the image
by the ``punchbowl.level1.destreak.correct_streaks`` function. Note that this requires that the image
is square; a check is carried out at the beginning of the correction to require this.


Applying correction
---------------------

The correction is carried out primarily in the ``punchbowl.level1.destreak.correct_streaks`` function:

.. autofunction:: punchbowl.level1.destreak.correct_streaks
    :no-index:

If you wish to incorporate this as a Prefect task in a custom pipeline,
using something like the ``punchbowl.level1.destreak.destreak_task`` is recommended.
