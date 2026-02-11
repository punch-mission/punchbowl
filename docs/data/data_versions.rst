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

Version 0j
----------
- Released February 10, 2026
- Paused realtime QuickPUNCH processing as new products are designed
- Added pB' to L3_PAM and L3_PTM images as a diagnostic
- Created new stray light modeling for WFI
- Update WFI flatfields
- Switched to a new cosmic ray despiking algorithm that is temporal instead of spatial
- Regenerated distortion models utilizing residual update for better alignment
- Introduced low-noise products L3_PAM and L3_CAM
- Fixed L3_PIM OBSLAYER values to reflect actual file contents
- Improved quicklook products

Version 0i
----------
- Released December 11, 2025
- Switched QuickPUNCH CTM to an f-corona subtracted image
- Improved intercalibration of WFI satellites with a new flat field
- Used a percentile band for F-corona model generation instead of only a threshold
- Improved image alignment with more attempts to solve
- Wrote an (approximate) uncertainty layer in F corona models, propagated it though F corona subtraction, and wrote the uncertainty layer for CIMs
- Changed to load NDCubes with CPDIS[1,2]A and DP[1,2]A set appropriately to avoid crashing Python
- Recorded WFI center locations in metadata for mosaics

Version 0h
----------
- Released October 29, 2025
- Fixed a double counting of pixel area that was making vignetting correction not work properly
- Recalculated PSF models to avoid artifacts and poor correction at edges of images
- Upgraded all level 0 files to 0h to support new BADPKTS keyword
- Fixed the observation layer keywrods for PFM and PIM
- Updated so distortion lookup table keywords are written for both WCS types
- Handled merging images properly where inf uncertainties and nan values are mixed
- Added noise reconstiution as part of square root decoding
- Properly set metadata for traceability in levels 2 and 3
- Changed to use powernorm for quicklook images
- Implemented new indexed background method for polarization
- Suppress saturated pixels by filling with neighborhood values

Version 0g
----------
- Released October 1, 2025
- Improved vignetting functions
- Improved PSF regularization resulting in a tighter output PSF
- NFI pointing "wobble" has been reduced via an improved optical distortion map
- Experimental L3 CIM and PIM images are now available, with preliminary F-corona subtraction. Starfield subtraction and low-noise mosaics are deferred to a later release.
- Known problems:

  + Images can contain blocky edges due to an incomplete application of the instrument mask
  + The OUTLIER flag is incorrectly set to True for many NFI images
  + Stray light subtraction for polarized images is computed inconsistently

Version 0f
----------
- Released September 16, 2025
- Reduced seams in L2 CTM/PTM mosaics
  - Benefiting from improved vignetting functions
  - The seams roll off smoothly from one image to another (`Pull request <https://github.com/punch-mission/punchbowl/pull/592>`_)
- The DATE header keyword is set correctly in L1 files (`Pull request <https://github.com/punch-mission/punchbowl/pull/586>`_)
- L2 and LQ CTM and PTM headers include "HAS_*" keywords indicating which imagers contributed to the mosaic. (`Pull request <https://github.com/punch-mission/punchbowl/pull/584>`_)
- L1 files contain a SPASE DOI (`Pull request <https://github.com/punch-mission/punchbowl/pull/583>`_)
- An error in polarization calculations was corrected

Version 0e
----------
- Released August 18, 2025
- Incorporate new outlier rejection
- Used alpha coefficients to inter-calibrate the WFIs
- Created automatic NFI flat-fielding module
- Sped up pointing refinement

Version 0d
----------
- Released August 8, 2025
- Fixed file provenance logging
- Addressed rolling stray light issues
- Dilated saturation in PSF correction more
- Turned on PTM processing
- Rigged up an automated reprocessing that is more ordered by time and dependencies
- Split Level 1 processing at stray light subtraction

Version 0c
----------
- Released July 24, 2025
- Included new PSF models
- Refined the pointing so it's more stable
- Implemented rolling stray light models
- Handled saturated pixels when building PSF model
- Handled mask when building PSF model
- Handled saturated pixels when correcting PSF
- Handled mask when correcting PSF
- Added lost in space pointing solver for when pointing isn't stable enough

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
