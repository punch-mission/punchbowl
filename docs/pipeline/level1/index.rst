Level 1
=======

The level 1 pipeline focuses on converting data from raw downlinked data to usable data in the spacecraft frame and field of view.

.. mermaid::

    graph LR;
    id1[initial uncertainty] --> id2[quartic fit];
    id2[quartic fit] --> id3[despike];
    id3[despike] --> id4[destreak];
    id4[destreak] --> id5[deficient pixel];
    id5[deficient pixel] --> id6[stray light];
    id6[stray light] --> id7[psf correction];
    id7[psf correction] --> id8[pointing];

.. toctree::
    :maxdepth: 1

    initial_uncertainty
    quartic_fit
    despike
    destreak
    deficient_pixel
    stray_light
    psf_correction
    pointing
