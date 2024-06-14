Deficient Pixel Replacement
=============================

Concept
---------

Out of the millions of pixels in an imaging sensor,
there can be quite a number with a deviating behavior that can range from absolutely dead (zero output signal),
to oversensitive (signal always saturated beyond a discernible signal).
If the deviation is strong enough and persistent in every image,
then this pixel value is in fact meaningless and will hamper image creation.
Thus, the deficient pixel correct algorithm uses a mask of pixels that are flagged as deficient and replaces them
with a value based on their neighborhood.

The value that is replaced will either be the mean or median of the neighborhood about the pixel. The neighborhood
starts off as just the immediate neighbors, but if too many of them are also deficient then it expands.

.. note::
  The deficient pixel map uses 0 to indicate a deficient pixel and 1 to indicate a normally behaving pixel.


Applying correction
---------------------

The correction is carried out primarily in the ``punchbowl.level1.deficient_pixel.remove_deficient_pixels`` function:

.. autofunction:: punchbowl.level1.deficient_pixel.remove_deficient_pixels
    :no-index:

If you wish to incorporate this as a Prefect task in a custom pipeline,
using something like the ``punchbowl.level1.deficient_pixel.remove_deficient_pixels_task`` is recommended.

Deriving the pixel mask
-------------------------
TODO
