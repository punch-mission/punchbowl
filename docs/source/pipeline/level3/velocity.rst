Velocity flow tracking
======================

Using PUNCH observations, flow tracking can be utilized to compute outflow velocities in the solar corona.

Concept
-------

A series of PUNCH mosaic observations can be projected into a common polar projection, from which a consideration of the cross-correlation over a window of time can extract the velocities of visible outflowing structure. These data are stored across all azimuths, and for at least four radial heights.

Applying method
---------------

Flow tracking is carried out in the ``punchbowl.level3.velocity.track_velocity`` function:

.. autofunction:: punchbowl.level3.velocity.track_velocity
    :no-index:

If you wish to incorporate this as a Prefect task in a custom pipeline,
using something like the ``punchbowl.level3.velocity.track_velocity`` is recommended.
