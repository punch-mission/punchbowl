PUNCH data structure
====================

Concept
-------
PUNCH data is structured using the `NDCube <https://docs.sunpy.org/projects/ndcube/en/stable/>`_ framework, which bundles an N-dimensional data array with a corresponding World Coordinate System (WCS) describing the spatial or spectral coordinates of the data itself. This allows for better integration with AstroPy software libraries, including coordinate and data reprojection or resampling. For primary Level 3 PUNCH data, connecting data with coordinates is critical given the relatively large field of view of the combined virtual PUNCH observatory. An NDCube object also allows for bundled metadata, uncertainty, masking, and visualization tools.

Uncertainty
-----------

The uncertainty is stored within the PUNCH NDCube data inside the pipeline as the absolute uncertainty in the data. When writing this data to a FITS file, the uncertainty is packed as the reciprocal of the ratio of the uncertainty to the data for each pixel. When unpacking these uncertainty values from a FITS file back into an NDCube object, this process is reversed, taking the inverse of the stored array and multiplying by the data array to restore the absolute uncertainty.

.. note::
    When reading data using the punchbowl framework, uncertainty is stored as the absolute uncertainty. When reading data using astropy FITS frameworks, uncertainty will appear as the reciprocal fractional uncertainty.
