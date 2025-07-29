Data Versions
=============

Every PUNCH file has a version. As the data processing code matures, it will be necessary and advantagous to create new versions of the data files. This document describes the versioning scheme and the state of each data version.

PUNCH data versions are either:
  - A number followed by a letter (e.g., 0c)
  - A number (e.g., 1).

Data versions sort lexicographically, so that version 1 comes before 1a, 1b, etc.

PUNCH data versions are not the same as:
  - The version of punchbowl software (e.g., 0.0.12.dev102+g2a26b4d) used to produce the data. The punchbowl version used to produce a file is documented internally in each file by the PIPEVRSN FITS header keyword.
  - The Level of the data (0, 1, 2, 3, Q, L), which designate the degree of processing required to produce the data. Level 0 data are data direct from the camera, whereas Level 3 data are highly processed.

The PUNCH mission (and the SDAC data repository) only supports the most recent version of the data.

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
