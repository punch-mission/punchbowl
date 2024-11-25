Polarization Resolution
=======================

Data enters level 3 processing in the MZP (minus 60 degrees, zero, and plus 60 degrees) representation of polarized light with respect to solar north in the unified mosaic frame. These observations can be transformed into brightness (B) and polarized brightness (pB).

Concept
-------

The polarization correction is primarily carried out by the `solpolpy <https://github.com/punch-mission/solpolpy>`_ framework, with a module within the punchbowl calling this code. See this code for full documentation.

Applying correction
-------------------

Polarization resolution is carried out in the ``punchbowl.level3.polarization.convert_polarization`` function:

.. autofunction:: punchbowl.level3.polarization.convert_polarization
    :no-index:

If you wish to incorporate this as a Prefect task in a custom pipeline,
using something like the ``punchbowl.level3.polarization.convert_polarization`` is recommended.
