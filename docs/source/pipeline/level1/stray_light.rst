Stray Light Removal
=======================

Stray light is light in an optical system, which does not follow the designed path for detection.
The light may be from the intended source, but follows paths other than that intended, or it may
be from a source other than the intended source. This light can limit the signal-to-noise ratio or
contrast ratio, by limiting how dark the system can be.

We estimate and remove the instrumental stray light within each PUNCH instrument. This is a
subtraction of the current additive stray light model,
which is developed from a time series of L1 data that is bootstrapped
with validation data acquired during on-orbit commissioning.
These are regenerated periodically to capture instrument changes.



Concept
---------

Separating instrumental stray light from the F-corona has been demonstrated with Solar and
Heliospheric Observatory (SOHO) Large Angle Spectrometric Coronagraph (LASCO) and with
STEREO/COR2 coronagraph data. It requires an instrumental roll to hold the stray light pattern
fixed while the F-corona rotates in the Field of View (FOV). PUNCH rolls once per orbit and the
same techniques previously used for stray light measurement with intermittent rolls on SOHO
and STEREO are used routinely for PUNCH.

Applying correction
---------------------

This correction is carried out by simply subtracting the stray light model using the
``punchbowl.level1.stray_light.remove_stray_light_task``.

.. autofunction:: punchbowl.level1.stray_light.remove_stray_light_task
    :no-index:

Deriving stray light model
---------------------------

TODO
