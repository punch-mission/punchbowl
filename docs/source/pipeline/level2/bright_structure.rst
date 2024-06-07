Bright Structure Identification
================================

Given a time sequence of images, this module identifies
"spikes" that exceed a threshold in a single frame.

Diffuse bright structures over the background F-corona are
identified and marked in the data using in-band "bad value" marking
(which is supported by the FITS standard). Data marking follows the
existing ZSPIKE temporal despiking algorithm to identify auroral
transients.

This is a voting algorithm based on my ZSPIKE.PRO from the 1990s,
which is available in the solarsoft distribution from Lockheed Martin.

ZSPIKE was originally used to identify cosmic rays, and
was adopted on STEREO for on-board despiking during exposure accumulation.
For this application ZSPIKE is ideal because it does not rely on the
spatial structure of spikes, only their temporal structure. Both cosmic
rays and, if present, high-altitude aurora are transient and easily
detected with ZSPIKE.

The algorithm assembles "votes" from the images
surrounding each one in a stream, to determine whether a particular pixel
is a good candidate for a temporal spike. If the pixel is sufficiently
bright compared to its neighbors in time, it is marked bad. "Bad values"
are stored in the DRP for file quality marking outlined in Module Quality
Marking.

There are two methods of identifying if a pixel is above a given threshold,
and therefore considered a spike. diff_method='abs' or 'sigma'.

diff_method 'abs' represents the absolute difference, and is the default.
If set, this is an absolute difference, in DN, required for a pixel to
'vote' its central value.  If the central value is this much higher than a
given voting value, then the central value is voted to be a spike.  If
it's this much lower, the veto count is incremented.

If diff_method 'sigma' is set if, then each pixel is treated as a
time series and the calculated sigma (RMS variation from the mean) of
the timeseries is used to calculate a difference threshold at each
location.

The threshold is the value over which a pixel is voted as a spike. The threshold
should be different depending diff_method.
