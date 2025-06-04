PUNCH data overview
====================

PUNCH is an imaging mission and most data from the mission are images.  Because the corona and solar wind are very faint
compared to the various backgrounds in the data, high photometric precision and many steps of processing are required.  This
means finding PUNCH data can be complicated.

There are four standard levels of processing for PUNCH data, numbered 0 through 3, plus two ancillary "levels" that are also produced but
are branches from the primary science data flow.  Each has a different collection of data
products.  Each data product is identified with a three-character code, that indicates the general type of product and (where relevant)
which spacecraft produced the data.

PUNCH is a polarimeter, so image data arrive in both unpolarized and polarized forms.  PUNCH represents polarization in two major ways:
first, as the quasi-Stokes parameters B, :sup:`⟂` pB, and :sup:`⟂` pB';
and second, as polarization triplets in the "M,Z,P" tri-polarizer system which uses three (real or virtual) polarizer
channels at M(inus) 60 degrees, Z(ero) degrees, and P(lus) 60 degrees relative to a reference angle in the
instrument or image plane.  Both systems are described by
`DeForest et al. 2022 <https://doi.org/10.3847/1538-4357/ac43b6>`_.

The four levels of processing are:

Level 0
-------

These are data direct from the PUNCH cameras, assembled into FITS files and merged with metadata from the spacecraft. The data are typically
square-root coded and losslessly compressed on board the spacecraft; L0 images have been decompressed into their original square-root
coded form, but the coding is preserved. The square-root coding is tuned to match the effective digital step size to the photon noise
level, across the dynamic range of the image.  To reconstruct direct camera values that are roughly proportional to photometric intensity,
you can examine the "ISSQRT" field in the header.  If "ISSQRT" has a nonzero value, then the radiance B of a particular pixel is given by

B = P * P / SCALE

where B is an approximation of the original value (in digitizer number units) from the camera, P is the value of a particular pixel,
and SCALE is the "SCALE" field in the header.

Raw camera pixel values are digitized at 16 bits.  NFI pixel values are sums of several camera frames, and can therefore have values
greater then 2 :sup:`16`.

Level 0 files have two-letter codes with a spacecraft number appended.  WFI1, WFI2, and WFI3 are assigned the numbers 1-3, and NFI is
assigned the number 4.  Polarized images have the codes "PMn", "PZn", and "PPn" where "n" is the spacecraft number.
The M, Z, and P refer to physical polarizer angles relative to camera-frame horizontal, though these are not calibrated at this level.
Clear images (taken with no polarizer in the beam) have the codes "CRn".
These sixteen codes (four image types and four spacecraft) form the bulk of the L0 dataset.

Level 1
-------

These are photometrically calibrated, conditioned data from the PUNCH cameras, maintained as separate data files from each
spacecraft.  The Level 1 data have similar naming scheme to Level 0, with the same product codes.  The data are destreaked, despiked,
flat-fielded, PSF-corrected, and stray-light subtracted; and have WCS fields with precise alignments derived from the in-image starfield.
The images are floating-point values in mean-solar-brightness units.

Level 2
-------

These are rectified NFI images and full-constellation mosaic trefoil images, resampled to output projection coordinates
(gnomonic projection for NFI and azimuthal-equidistant projection for the mosaics).  Polarization is given in the
virtual-polarizer M,Z,P system.  The images are intended to be photometric, astrometrically regularized images of the
celestial sphere with all light sources included.

The product codes are "PTM" for polarized trefoil mosaics, "CTM" for clear trefoil mosaics, "PNN" for polarized NFI images, and "CNN" for
clear NFI images.

Level 3
-------

Level 3 products are background-subtracted and are intended to be usable directly as coronal and solar-wind images.  The basic product
codes are similar to the Level 2 codes, but include long-term average products ("PAM", "PAN", "CAM", and "CAN") that fill in the entire circular
FOV by averaging across 32 minutes of PUNCH data acquisition. In addition, a wind-flow product "VAM" describes derived solar-wind motion.

Level Q: QuickPUNCH
-------------------

These are images, derived from the Level 2 data, that are intended to be useful for space weather forecasting.  They use a time-asymmetric
background modeling scheme that permits low-latency production.  They're produced for NOAA's space weather forecasting infrastructure,
but are also made available to other users.

Level L: QuickLook
------------------

These images are intended for viewing and are published in JPEG2000 (.jp2) format.  They're derived using the same time-asymmetric background as
QuickPUNCH.
