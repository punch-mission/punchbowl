Starfield Removal
=================

Stellar light visible in PUNCH observations can be characterized and removed once the data has been processed and projected into the same frame.

Concept
-------

By gathering observations and projecting the data into a common celestial frame, the starfield signal can be gathered and estimated. For a given observation in the helioprojective frame, this map of stellar signal can be projected back and subtracted, revealing the desired coronal component of the input signal.

Applying correction
-------------------

Image resampling is carried out in the ``punchbowl.level3.stellar.subtract_starfield_background_task`` function:

.. autofunction:: punchbowl.level3.stellar.subtract_starfield_background_task
    :no-index:

If you wish to incorporate this as a Prefect task in a custom pipeline,
using something like the ``punchbowl.level3.stellar.subtract_starfield_background_task`` is recommended.

Deriving starfield model
------------------------

Starfield models are generated from data using the `remove_starfield <https://github.com/svank/remove_starfield>`_ package, where further documentation can be found.
