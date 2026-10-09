"""Constant values used in PUNCHBOWL project."""
from enum import IntEnum

#: Data version: Current data version
CURRENT_DATA_VERSION = "v0l"

#: punch resolution
ORIGINAL_PUNCH_RESOLUTION = 2048

# --- NFI processing related -------------------------------------------------------------------------------------------
TINY = 1.0e-4

# --- Glint location parameters for NFI Glint masking ------------------------------------------------------------------
# Used by: `generate_glint_mask()` (which is also called by `remove_nfi_stray_light()`) in nfi_dynamic_stray_light.py
#
# These values were technically determined experimentally to mask out the glint spheres in the NFI images as
# consistently as possible.
# Therefore the parameters in generate_glint_mask are technically modifiable for fine-tuning, but are not expected to
# vary from these values by much.
GLINT_SPHERE1_CENTER = (540,790)
GLINT_SPHERE2_CENTER = (540,1210)
GLINT_SPHERE_RADIUS = 375
GLINT_MASK_BOTTOM_CUT_OFF = 250

# --- Straylight Kernel generation related constants -------------------------------------------------------------------
# Center of the kernel---the default values are the center of the occulted region, which isn't necessarily the
# center of the donut of stray light
KERNEL_CENTER_X = 1014.50355056 - 1
KERNEL_CENTER_Y = 1037.37339562 - 1

# --- Phases of images -------------------------------------------------------------------------------------------------
class ImagePhase(IntEnum):
    """
    Represents phases of image collection.

    1 through 3 are the first set of polarized.
    4 is the clear.
    4 through 7 are the second set of polarized.
    """

    POLARIZED_PP_PHASE_1 = 1
    POLARIZED_PZ_PHASE_1 = 2
    POLARIZED_PM_PHASE_1 = 3
    CLEAR = 4
    POLARIZED_PP_PHASE_2 = 5
    POLARIZED_PZ_PHASE_2 = 6
    POLARIZED_PM_PHASE_2 = 7
