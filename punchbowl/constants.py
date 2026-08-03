"""Constant values used in PUNCHBOWL project."""

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
