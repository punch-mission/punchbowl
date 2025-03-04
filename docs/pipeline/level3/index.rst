Level 3
=======

The level 3 pipeline focuses on removing the f-corona and starfield signal from the data, and remixing polarization into brightness and polarized brightness. A flow tracking module exists for extracting velocity data. A separate module and flow exists for generating low-noise combined mosaics.

.. mermaid::

    graph LR;
    id1[f-corona subtraction] --> id2[starfield subtraction];
    id2[starfield subtraction] --> id3[polarization remixing];
    id3[polarization remixing] --> id4[velocity flow tracking];
    id3[polarization remixing] --> id5[low noise product generation];

.. toctree::
    :maxdepth: 1

    f_corona
    starfield
    polarization
    velocity
    low_noise
