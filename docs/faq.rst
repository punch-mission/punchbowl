FAQ
===


- **The FITS file says the units for PUNCH data is 2.009e+07 W/(m2 sr); I thought it was in B/Bsun?**
    Sunpy wants physical units. This is Bsun mean solar brightness in physical units. Also known as Mean Solar Brightness -- MSB.

- **Are pixels in degrees elongation?**
    Every pixel has a corresponding heliographic latitude/longitude which is degrees elongation.

- **What are the three layers in PAM files?**
    The three layers are tB, pB, and pB'.
    The pB we use is also called "^⟂ pB", pronounced "perp-pB":
    it is analogous to Stokes Q, but with orientation that varies across the focal plane (positive representing polarization perpendicular to the local solar-radial direction on the image plane).
    The "pB'" bears the same relationship to pB that Stokes U bears to Stokes Q: it measures polarization diagonal to the local solar-radial direction.
    The pB' layer is there as a measure of background-removal effectiveness: it should be identically zero,
    in the ideal world, and deviation from that indicates work still to do in the polarization code or the background subtraction.
