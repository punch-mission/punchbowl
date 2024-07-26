
import numpy as np
from ndcube import NDCube
from prefect import flow, get_run_logger
from regularizepsf import CoordinatePatchCollection, simple_psf

from punchbowl.level1.alignment import align_task
from punchbowl.level1.deficient_pixel import remove_deficient_pixels_task
from punchbowl.level1.despike import despike_task
from punchbowl.level1.destreak import destreak_task
from punchbowl.level1.initial_uncertainty import update_initial_uncertainty_task
from punchbowl.level1.psf import correct_psf_task
from punchbowl.level1.quartic_fit import perform_quartic_fit_task
from punchbowl.level1.stray_light import remove_stray_light_task
from punchbowl.level1.vignette import correct_vignetting_task
from punchbowl.util import load_image_task, output_image_task


@flow(validate_parameters=False)
def generate_psf_model_core_flow(input_filepaths: []) -> NDCube:  # noqa: ARG001
    """Generate PSF model."""
    # Define the parameters and image to use
    psf_size = 32
    patch_size = 256
    target_fwhm = 3.25
    image_fn = "data/DASH.fits"

    # Define the target PSF
    center = patch_size / 2
    sigma = target_fwhm / 2.355

    @simple_psf
    def target(x, y, x0=center, y0=center, sigma_x=sigma, sigma_y=sigma) -> np.ndarray:  # noqa: ANN001
        return np.exp(-(np.square(x - x0) / (2 * np.square(sigma_x)) + np.square(y - y0) / (2 * np.square(sigma_y))))

    target_evaluation = target(*np.meshgrid(np.arange(patch_size), np.arange(patch_size)))
    coordinate_patch_collection = CoordinatePatchCollection.find_stars_and_average(
        [image_fn], psf_size, patch_size)
    array_corrector = coordinate_patch_collection.to_array_corrector(target_evaluation)  # noqa: F841

    return NDCube(data=None, wcs=None, meta=None)  # TODO: convert array_corrector into PUNCHData


@flow(validate_parameters=False)
def level1_core_flow(
    input_data: str | NDCube,
    quartic_coefficient_path: str | None = None,
    vignetting_function_path: str | None = None,
    stray_light_path: str | None = None,
    deficient_pixel_map_path: str | None = None,
    output_filename: str | None = None,
) -> list[NDCube]:
    """Core flow for level 1."""
    logger = get_run_logger()

    logger.info("beginning level 1 core flow")

    data = load_image_task(input_data) if isinstance(input_data, str) else input_data

    data = update_initial_uncertainty_task(data)
    data = perform_quartic_fit_task(data, quartic_coefficient_path)
    data = despike_task(data)  # TODO: allow configuration of the run with different despike options
    data = destreak_task(data)
    data = correct_vignetting_task(data, vignetting_function_path)
    data = remove_deficient_pixels_task(data, deficient_pixel_map_path)
    data = remove_stray_light_task(data, stray_light_path)
    data = correct_psf_task(data)
    data = align_task(data)
    logger.info("ending level 1 core flow")

    if output_filename is not None:
        output_image_task(data, output_filename)
    return [data]
