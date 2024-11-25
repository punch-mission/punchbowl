Data Product Codes
====================

PUNCH data products are organized into data processing levels from Level 0 (raw camera data) to Level 3 (calibrated science data). Within and across levels distinct data products and calibration files are marked with a unique identifying product code. For data in the spacecraft frame, each spacecraft is marked with a unique numeral identity - 1,2,3 for each WFI, or 4 for NFI.

Data Product Codes
------------------

.. list-table::
   :header-rows: 1

   * - Level
     - Code
     - Description
   * - 0
     - PM1, PM2, PM3, PM4, PZ1, PZ2, PZ3, PZ4, PP1, PP2, PP3, PP4, CR1, CR2, CR3, CR4
     - Science images in the standard polarization (PM, PZ, PP) and clear (CR) states
   * - 0
     - PX1, PX2, PX3, PX4
     - Science images in a nonstandard polarization state
   * - 0
     - ST
     - STEAM data packet (CSV format)
   * - 1
     - PM1, PM2, PM3, PM4, PZ1, PZ2, PZ3, PZ4, PP1, PP2, PP3, PP4, CR1, CR2, CR3, CR4
     - Science images in the standard polarization (PM, PZ, PP) and clear (CR) states (photometrically calibrated)
   * - 2
     - PTM
     - Pol. science mosaics (Trefoil) in output coordinates, resolved into MZP pol. triplets, and uncertainty layer
   * - 2
     - CTM
     - Clear science mosaics (Trefoil) in output coordinates, resolved into image and uncertainty layer
   * - 2
     - PNN
     - Pol. NFI images in output coordinates, resolved into MZP pol. triplets, and uncertainty layer
   * - 2
     - CNN
     - Clear NFI images in output coordinates, resolved into image, and uncertainty layer
   * - Q
     - CNN
     - QuickPUNCH NFI images
   * - Q
     - CTM
     - QuickPUNCH Mosaic images (5.4–80 Rsun)
   * - L
     - CNN
     - Quicklook Clear NFI images
   * - L
     - PNN
     - Quicklook Polarized NFI images
   * - L
     - CTM
     - Quicklook Clear Mosaic images
   * - L
     - PTM
     - Quicklook Polarized Mosaic images
   * - 3
     - CAM
     - Clear low-noise science mosaic, bkg-sub & resolved into B & uncertainty layer
   * - 3
     - PAN
     - Polarized low-noise NFI science image, bkg-sub & resolved into B, pB, & uncertainty layer
   * - 3
     - CAN
     - Clear low-noise NFI science image, bkg-sub & resolved into B & uncertainty layer
   * - 3
     - PTM
     - Polarized K PUNCH science mosaics (Trefoil), bkg-sub & resolved into B, pB, & uncertainty layer
   * - 3
     - CTM
     - Clear K science mosaics (Trefoil), bkg-sub & resolved into B & uncertainty layer
   * - 3
     - PNN
     - Polarized K NFI science image, bkg-sub & resolved into B, pB, & uncertainty layer
   * - 3
     - CNN
     - Clear K NFI science image, bkg-sub & resolved into B & uncertainty layer
   * - 3
     - VAM
     - Mosaic derived wind velocity maps extracted from MP’s: 1440 pos. angles at various altitudes
   * - 3
     - VAN
     - NFI derived wind velocity maps extracted from MP’s: 1440 pos. angles at various altitudes
   * - 3
     - PAM
     - Polarized low-noise science mosaic, bkg-sub & resolved into B, pB, & uncertainty layer


Calibration Product Codes
-------------------------

.. list-table::
   :header-rows: 1

   * - Level
     - Code
     - Description
   * - 0
     - DK1, DK2, DK3, DK4, DY1, DY2, DY3, DY4
     - Calibration Images: polarizer in dark pos.; stim lamp off (DK) or on (DY)
   * - 0
     - OV1, OV2, OV3, OV4
     - Calibration Image: CCD over-scan
   * - 0
     - XI1, XI2, XI3, XI4
     - Experimental image (no set parameters; variable crop)
   * - 1
     - BD1, BD2, BD3, BD4
     - Calibration: Deficient Pixel (Boolean) Map
   * - 1
     - FQ1, FQ2, FQ3, FQ4
     - Calibration: Flat-field parameters (quartic polynomial coefficients), by pixel
   * - 1
     - GM1, GM2, GM3, GM4, GZ1, GZ2, GZ3, GZ4, GP1, GP2, GP3, GP4, GR1, GR2, GR3, GR4
     - Calibration: Vignetting functions for the standard polarization (GM, GZ, GP) and clear (GR) states
   * - 1
     - SM1, SM2, SM3, SM4, SZ1, SZ2, SZ3, SZ4, SP1, SP2, SP3, SP4, SR1, SR2, SR3, SR4
     - Calibration: Instrumental additive stray light model for the standard polarization (SM, SZ, SP) and clear (SR) states
   * - 1
     - RM1, RM2, RM3, RM4, RZ1, RZ2, RZ3, RZ4, RP1, RP2, RP3, RP4, RC1, RC2, RC3, RC4
     - Calibration: Point Spread Function model for the standard polarization and clear states
   * - Q
     - CFN
     - QuickPUNCH NFI images F corona model
   * - Q
     - CFM
     - QuickPUNCH Mosaic images (5.4–80 Rsun) F corona model
   * - 3
     - PFM
     - Polarized mosaic F corona model, resolved into MZP pol. triplets, and uncertainty layer (from MP’s)
   * - 3
     - CFM
     - Clear mosaic F corona model, resolved into image and uncertainty layer (from MC’s)
   * - 3
     - PFN
     - Polarized NFI F-corona model, resolved into MZP pol. triplets, and uncertainty layer (from NP’s)
   * - 3
     - CFN
     - Clear NFI F-corona model, resolved into image and uncertainty layer (from NC’s)
   * - 3
     - PSM
     - Polarized mosaic stellar model, resolved into MZP pol. triplets, and uncertainty layer (from MP’s)
   * - 3
     - CSM
     - Clear mosaic stellar model, resolved into image and uncertainty layer (from MC’s)
   * - 3
     - PSN
     - Polarized NFI stellar model, resolved into MZP pol. triplets, and uncertainty layer (from MP’s)
   * - 3
     - CSN
     - Clear NFI stellar model, resolved into image and uncertainty layer (from MC’s)
