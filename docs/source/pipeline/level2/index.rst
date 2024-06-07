Level 2
========

The level 2 pipeline focuses on the steps to resolve polarization and merge images together. It consists of a
few modules:


.. mermaid::

    graph LR;
    id1[polarization resolution] --> id2[image resampling];
    id2[image resampling] --> id3[bright structure identification];
    id3[bright structure identification] --> id4[image merging];


.. toctree::

    polarization
    image_resample
    bright_structure
    image_merging
