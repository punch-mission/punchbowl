Data Versions
=============

Every PUNCH file has a version. As the data processing code matures, it will be necessary and advantageous to create new versions of the data files. This document describes the versioning scheme and each data version.

PUNCH data versions are either:
  - A number followed by a letter (e.g., 0c)
  - A number (e.g., 1).

Data versions sort lexicographically, so that version 1 comes before 1a, 1b, etc. The data version is given at the end of each FITS file name, preceded by a "_v". As an example, file PUNCH_L0_CR2_20250501235153_v0.fits has a data version of 0.

PUNCH data versions are not the same as:
  - The version of punchbowl software (e.g., 0.0.12.dev102+g2a26b4d) used to produce the data. The punchbowl version used to produce a file is documented internally in each file by the PIPEVRSN FITS header keyword.
  - The Level of the data (0, 1, 2, 3, Q, L), which designate the degree of processing required to produce the data. Level 0 data are data direct from the camera, whereas Level 3 data are highly processed.

The PUNCH mission and the SDAC data repository only supports the most recent version of the data. No backwards compatibility is assured. Especially during the early mission phase, users should be careful that the data processing implemented to date suits their needs, and that they are using the most recent version of the data. If in doubt, users are advised to `start a discussion on GitHub <https://github.com/punch-mission/punchbowl/discussions/new/choose>`_ for questions regarding the suitability of the current data for their needs. Users can expect that an increment of the integer portion of the version will designate a more significant change than an increment of the letter portion of the version.

A history of data version releases is given below.

Version 0b
----------
- Released June 1, 2025
- Small metadata improvements from 0a
- Includes Level 1 and Level Q products

Version 0a
-----------
- Released May 14, 2025
- Initial version released during commissioning
- Only Level 0 products
