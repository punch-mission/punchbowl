PUNCHData object
=================

Concept
---------
The PUNCHData object provides a layer atop a typical NDCube object for PUNCH-specific functions. The object includes a primary data array, a WCS describing that data, metadata, and an uncertainty array describing that primary data. Data can be loaded either through this particular object, or through standard FITS readers.

Uncertainty
-------------
The uncertainty is stored within the PUNCHData object as a floating-point value from 0-1, describing the relative uncertainty of that pixel - 0 being complete certainty and 1 being complete uncertainty. When written to file, these are stored as 8-bit integer values using the HDU scaling feature. Most FITS readers (AstroPy tested) will reconstitute these to the original range of 0-1.
