import numpy as np
import regularizepsf
from regularizepsf.image_processing import _find_patches
from scipy.ndimage import binary_dilation, binary_erosion, label

from punchbowl.level1.psf import generate_projected_psf


def correct_psf_per_image(cube, psf_size=64, proj_psf: regularizepsf.ArrayPSF = None, debug: bool = False,
                          saturation_threshold: float = np.inf, star_min=0, star_max=np.inf):
    patches = _find_patches(cube.data[624:-624, 624:-624].copy(), 3, None, 1, psf_size, 0,
                            saturation_threshold=saturation_threshold, image_mask=None,
                            star_minimum=star_min, star_maximum=star_max)
    patch_stack = np.array(list(patches.values()))
    center_vals = patch_stack[:, psf_size // 2, psf_size // 2]
    patch_stack = np.array([patch / center_val for patch, center_val in zip(patch_stack, center_vals)])

    cleaned_patch_stack = np.zeros_like(patch_stack)
    for i, patch in enumerate(patch_stack):
        patch_background = regularizepsf.image_processing.calculate_background(patch)
        patch -= patch_background

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
    average_patch = np.nanmean(cleaned_patch_stack, axis=0)
    average_patch[average_patch <= 1E-3] = 0
    average_patch /= np.sum(average_patch)

    if proj_psf is None:
        proj_psf = generate_projected_psf(cube.wcs, psf_size, star_gaussian_sigma=2 / 2.366)
    new_values = proj_psf.values.copy()
    for i in range(new_values.shape[0]):
        new_values[i] = average_patch
    source = regularizepsf.ArrayPSF(regularizepsf.util.IndexedCube(proj_psf.coordinates, new_values))
    t = regularizepsf.ArrayPSFTransform.construct(source, proj_psf, 0.7, 0.1)
    result = t.apply(cube.data.copy())
    if debug:
        return result, source, proj_psf
    return result
