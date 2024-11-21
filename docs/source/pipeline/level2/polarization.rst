Polarization Resolution
=========================

Each PUNCH spacecraft observes polarized light in three polarizer wheel states, denoted M, Z, and P (minus 60 degrees, zero, and plus 60 degrees). These measurements are in the spacecraft frame, however. To merge these observations into a unified virtual field of view, these polarization data must be represented in a unified MZP frame with respect to the solar frame.

Concept
-------

The polarization correction is primarily carried out by the `solpolpy <https://github.com/punch-mission/solpolpy>`_ framework, with a module within the punchbowl calling this code. See this code for full documentation.

Applying correction
-------------------

Polarization resolution is carried out in the ``punchbowl.level2.polarization.resolve_polarization`` function:

.. autofunction:: punchbowl.level2.polarization.resolve_polarization
    :no-index:

If you wish to incorporate this as a Prefect task in a custom pipeline,
using something like the ``punchbowl.level2.polarization.resolve_polarization_task`` is recommended.
