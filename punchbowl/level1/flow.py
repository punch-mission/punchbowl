import numpy as np
from ndcube import NDCube
from prefect import flow, get_run_logger
from regularizepsf import ArrayCorrector, CoordinatePatchCollection, simple_psf

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
def generate_psf_model_core_flow(input_filepaths: [str],
                                 psf_size: int = 32,
                                 patch_size: int = 256,
                                 target_fwhm: float = 3.25) -> ArrayCorrector:
    """Generate PSF model."""
    # Define the target PSF as a symmetric Gaussian
    center = patch_size / 2
    sigma = target_fwhm / 2.355

    @simple_psf
    def target(x, y, x0=center, y0=center, sigma_x=sigma, sigma_y=sigma) -> np.ndarray:  # noqa: ANN001
        return np.exp(-(np.square(x - x0) / (2 * np.square(sigma_x)) + np.square(y - y0) / (2 * np.square(sigma_y))))

    target_evaluation = target(*np.meshgrid(np.arange(patch_size), np.arange(patch_size)))

    # Build the PSF model using the target PSF
    coordinate_patch_collection = CoordinatePatchCollection.find_stars_and_average(input_filepaths,
                                                                                   psf_size,
                                                                                   patch_size)
    return coordinate_patch_collection.to_array_corrector(target_evaluation)


@flow(validate_parameters=False)
def level1_core_flow(
    input_data: str | NDCube,
    gain: float = 4.3,
    bias_level: float = 100,
    dark_level: float = 55.81,
    read_noise_level: float = 17,
    bitrate_signal: int = 16,
    quartic_coefficient_path: str | None = None,
    despike_unsharp_size: int = 3,
    despike_method: str = "convolve",
    despike_alpha: float = 1.0,
    despike_dilation: int = 0,
    exposure_time: float = 49 * 1000,
    readout_line_time: float = 163/2148,
    reset_line_time: float = 163/2148,
    vignetting_function_path: str | None = None,
    stray_light_path: str | None = None,
    deficient_pixel_map_path: str | None = None,
    deficient_pixel_method: str = "median",
    deficient_pixel_required_good_count: int = 3,
    deficient_pixel_max_window_size: int = 10,
    psf_model_path: str | None = None,
    output_filename: str | None = None,
) -> list[NDCube]:
    """Core flow for level 1."""
    logger = get_run_logger()

    logger.info("beginning level 1 core flow")

    data = load_image_task(input_data) if isinstance(input_data, str) else input_data

    data = update_initial_uncertainty_task(data,
                                           bias_level=bias_level,
                                           dark_level=dark_level,
                                           gain=gain,
                                           read_noise_level=read_noise_level,
                                           bitrate_signal=bitrate_signal,
                                           )
    data = perform_quartic_fit_task(data, quartic_coefficient_path)
    data = despike_task(data,
                        unsharp_size=despike_unsharp_size,
                        method=despike_method,
                        alpha=despike_alpha,
                        dilation=despike_dilation)
    data = destreak_task(data,
                         exposure_time=exposure_time,
                         reset_line_time=reset_line_time,
                         readout_line_time=readout_line_time)
    data = correct_vignetting_task(data, vignetting_function_path)
    data = remove_deficient_pixels_task(data,
                                        deficient_pixel_map_path,
                                        required_good_count=deficient_pixel_required_good_count,
                                        max_window_size=deficient_pixel_max_window_size,
                                        method=deficient_pixel_method)
    data = remove_stray_light_task(data, stray_light_path)
    data = correct_psf_task(data, psf_model_path)
    data = align_task(data)
    logger.info("ending level 1 core flow")

    if output_filename is not None:
        output_image_task(data, output_filename)
    return [data]


if __name__ == "__main__":
    from pathlib import Path
    level1_core_flow("/Users/jhughes/Desktop/data/gamera_mosaic_jan2024/synthetic_l0/PUNCH_L0_PM1_20240620001200_v1.fits",
                     vignetting_function_path=Path("/Users/jhughes/Desktop/repos/punchbowl/test_run/PUNCH_L1_GM1_20240817174727_v1.fits"),
                     despike_unsharp_size=1,
                     despike_alpha=3,
                     despike_method="median",
                     psf_model_path="/Users/jhughes/Desktop/repos/punchbowl/test_run/synthetic_forward_psf.h5",
                     output_filename="/Users/jhughes/Desktop/repos/punchbowl/test_run/test_out4.fits")
