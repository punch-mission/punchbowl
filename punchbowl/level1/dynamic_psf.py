import os
from pathlib import Path

import cv2
import numpy as np
import regularizepsf
from regularizepsf.image_processing import _find_patches
from scipy.ndimage import binary_dilation, binary_erosion, label, shift

from punchbowl.data.punchcube import PUNCHCube
from punchbowl.prefect import get_logger, punch_task
from punchbowl.util import DataLoader


def correct_psf_per_image(
    cube: PUNCHCube,
    target_psf: regularizepsf.ArrayPSF,
    psf_size: int = 64,
    saturation_threshold: float = np.inf,
    star_threshold: float = 3,
    star_min: float = 0,
    star_max: float = np.inf,
    box: tuple[int, int, int, int] = (1280- 300, 1280 + 300, 1280 - 300, 1280 + 300),
    alpha: float= 1.05,
    epsilon: float = 0.02,
) -> PUNCHCube:
    """
    Correct the dynamic point spread function for each image.

    This works by sampling a region of the image, in the specified box, and building a PSF model on the fly.

    Parameters
    ----------
    cube : PUNCHCube
        cube to correct
    target_psf : regularizepsf.ArrayPSF
        the PSF that is used as the target for the transform to correct each image
    psf_size : int
        size of the PSF model in pixels
    saturation_threshold : float
        when finding patches, ignore any pixels above this brightness
    star_threshold : float
        how many sigma above the noise does a signal need to be to consider it a star
    star_min : float
        the minimum brightness to consider for stars
    star_max : float
        the maximum brightness to consider for stars
    box : tuple[int, int, int, int]
        the lower x, higher x, lower y, higher y coordinates the form a box in the original image to use for sampling
    alpha: float
        the alpha parameter used in regularization, see the Coma Off It Paper
    epsilon: float
        the epsilon parameter used in regularization, see the Coma Off It Paper

    Returns
    -------
    PUNCHCube
        a PUNCHCube with corrected PSF

    """
    box_low_x, box_high_x, box_low_y, box_high_y = box

    patches = _find_patches(
        cube.data[box_low_y:box_high_y, box_low_x:box_high_x].copy(),
        star_threshold,
        None,  # no star mask will be masked through
        1,  # no interpolation will be carried out
        psf_size,
        0,  # this is the image index that gets passed through, it's irrelevant here
        saturation_threshold=saturation_threshold,
        image_mask=None,
        star_minimum=star_min,
        star_maximum=star_max,
    )
    patch_stack = np.array(list(patches.values()))
    center_vals = patch_stack[:, psf_size // 2, psf_size // 2]
    patch_stack = np.array([patch / center_val for patch, center_val
                            in zip(patch_stack, center_vals, strict=True)])

    # we remove the background brightness from the patches since the stars are embedded in a bright background
    cleaned_patch_stack = np.zeros_like(patch_stack)
    for i, patch in enumerate(patch_stack):
        patch_background = regularizepsf.image_processing.calculate_background(patch)
        patch -= patch_background  # noqa: PLW2901, we're fine mutation the data

        patch[patch == 0] = np.nan

        patch_central_value = patch[patch.shape[0] // 2, patch.shape[1] // 2]
        this_value_mask = patch < (0.005 * patch_central_value)
        this_value_mask = binary_erosion(this_value_mask, border_value=1)

        patch[this_value_mask] = np.nan

        patch_zeroed = np.copy(patch)
        patch_zeroed[~np.isfinite(patch_zeroed)] = 0

        patch_labeled = label(patch_zeroed)[0]
        psf_core_mask = patch_labeled == patch_labeled[patch_labeled.shape[0] // 2, patch_labeled.shape[1] // 2]

        psf_core_mask = binary_dilation(psf_core_mask)

        patch_corrected = patch_zeroed * psf_core_mask
        patch_corrected = patch_corrected / np.nansum(patch_corrected)
        cleaned_patch_stack[i] = patch_corrected

    # average the patches directly and normalize so that they sum to 1
    average_patch = np.abs(np.nanmedian(cleaned_patch_stack, axis=0))
    average_patch /= np.sum(average_patch)

    # the patches may not be centered which can result in a pointing misalignment
    # we carefully make sure the candidate patches are centered by shifting them into the middle of the patch
    moments = cv2.moments(average_patch)
    average_patch = shift(average_patch, (psf_size // 2 - moments["m01"], psf_size // 2 - moments["m10"]))

    # copy the target PSF over and create a transform model, then apply it
    new_values = target_psf.values.copy()
    for i in range(new_values.shape[0]):
        new_values[i] = average_patch.copy()
    source = regularizepsf.ArrayPSF(regularizepsf.util.IndexedCube(target_psf.coordinates, new_values))
    t = regularizepsf.ArrayPSFTransform.construct(source, target_psf, alpha, epsilon)
    result = t.apply(cube.data.copy(), normalization_coefficient=(8 / 9) ** 2)

    # mutate the cube and return it
    cube.data[...] = result
    return cube


@punch_task
def correct_dynamic_psf_task(
    data_object: PUNCHCube,
    target_psf_path: str | DataLoader | None = None,
) -> PUNCHCube:
    """
    Prefect Task to correct the dynamic PSF of an image.

    Parameters
    ----------
    data_object : PUNCHCube
        data to operate on
    target_psf_path : str | DataLoader | None
        path to the target PSF (an ArrayPSF) the image is transformed to; if None, no correction is applied

    Returns
    -------
    PUNCHCube
        modified version of the input with the dynamic PSF corrected

    """
    if target_psf_path is not None:
        if isinstance(target_psf_path, DataLoader):
            target_psf = target_psf_path.load()
            target_psf_path = target_psf_path.src_repr()
        else:
            target_psf = regularizepsf.ArrayPSF.load(Path(target_psf_path))
        data_object = correct_psf_per_image(data_object, target_psf)
        data_object.meta.history.add_now("LEVEL1-correct_dynamic_psf",
                                         f"Dynamic PSF corrected to {os.path.basename(target_psf_path)} target")
    else:
        data_object.meta.history.add_now("LEVEL1-correct_dynamic_psf",
                                         "Empty target path so no dynamic PSF correction applied")
        logger = get_logger()
        logger.info("No target PSF path so dynamic PSF correction is skipped")
    return data_object
