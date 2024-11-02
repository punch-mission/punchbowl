from pathlib import Path
from collections.abc import Callable

import astropy.units as u
import numpy as np
from ndcube import NDCube
from prefect import flow, get_run_logger
from regularizepsf import ArrayPSFBuilder, ArrayPSFTransform, simple_functional_psf
from regularizepsf.util import calculate_covering

from punchbowl.data import NormalizedMetadata
from punchbowl.data.units import dn_to_msb
from punchbowl.level1.alignment import align_task
from punchbowl.level1.deficient_pixel import remove_deficient_pixels_task
from punchbowl.level1.despike import despike_task
from punchbowl.level1.destreak import destreak_task
from punchbowl.level1.initial_uncertainty import update_initial_uncertainty_task
from punchbowl.level1.psf import correct_psf_task
from punchbowl.level1.quartic_fit import perform_quartic_fit_task
from punchbowl.level1.sqrt import decode_sqrt_data
from punchbowl.level1.stray_light import remove_stray_light_task
from punchbowl.level1.vignette import correct_vignetting_task
from punchbowl.util import load_image_task, output_image_task


@flow(validate_parameters=False)
def generate_psf_model_core_flow(input_filepaths: [str],
                                 alpha: float = 2.0,
                                 epsilon: float = 0.3,
                                 image_shape: (int, int) = (2048, 2048),
                                 psf_size: int = 32,
                                 target_fwhm: float = 3.25) -> ArrayPSFTransform:
    """Generate PSF model."""
    # Define the target PSF as a symmetric Gaussian
    center = psf_size / 2
    sigma = target_fwhm / 2.355
    @simple_functional_psf
    def target(x, y, x0=center, y0=center, sigma_x=sigma, sigma_y=sigma) -> np.ndarray:  # noqa: ANN001
        return np.exp(-(np.square(x - x0) / (2 * np.square(sigma_x)) + np.square(y - y0) / (2 * np.square(sigma_y))))

    image_psf = ArrayPSFBuilder(psf_size).build(input_filepaths)
    coords = calculate_covering(image_shape, psf_size)
    return ArrayPSFTransform.construct(image_psf, target.as_array_psf(coords, psf_size), alpha, epsilon)


@flow(validate_parameters=False)
def level1_core_flow(
    input_data: list[str] | list[NDCube],
    gain: float = 4.9,
    bias_level: float = 100,
    dark_level: float = 55.81,
    read_noise_level: float = 17,
    bitrate_signal: int = 16,
    quartic_coefficient_path: str | None = None,
    despike_unsharp_size: int = 1,
    despike_method: str = "median",
    despike_alpha: float = 3.0,
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
    alignment_mask: Callable | None = None,
    output_filename: str | None = None,
) -> list[NDCube]:
    """Core flow for level 1."""
    logger = get_run_logger()

    logger.info("beginning level 1 core flow")

    output_data = []
    for i, this_data in enumerate(input_data):
        data = load_image_task(this_data) if isinstance(this_data, str) else this_data

        data = decode_sqrt_data(data)
        data = update_initial_uncertainty_task(data,
                                               bias_level=bias_level,
                                               dark_level=dark_level,
                                               gain=gain,
                                               read_noise_level=read_noise_level,
                                               bitrate_signal=bitrate_signal,
                                               )
        data = perform_quartic_fit_task(data, quartic_coefficient_path)

        if data.meta["OBSCODE"].value == "4":
            scaling = {"gain": 4.9 * u.photon / u.DN,
                       "wavelength": 530. * u.nm,
                       "exposure": 49 * u.s,
                       "aperture": 49.57 * u.mm ** 2}
        else:
            scaling = {"gain": 4.9 * u.photon / u.DN,
                       "wavelength": 530. * u.nm,
                       "exposure": 49 * u.s,
                       "aperture": 34 * u.mm ** 2}
        data.data[:, :] = np.clip(dn_to_msb(data.data[:, :], data.wcs, **scaling), a_min=0, a_max=None)
        data.uncertainty.array[:, :] = dn_to_msb(data.uncertainty.array[:, :], data.wcs, **scaling)

        data = despike_task(data,
                            unsharp_size=despike_unsharp_size,
                            method=despike_method,
                            alpha=despike_alpha,
                            dilation=despike_dilation)
        data = destreak_task(data,
                             exposure_time=exposure_time,
                             reset_line_time=reset_line_time,
                             readout_line_time=readout_line_time)
        data = correct_vignetting_task(data, Path(vignetting_function_path))
        data = remove_deficient_pixels_task(data,
                                            deficient_pixel_map_path,
                                            required_good_count=deficient_pixel_required_good_count,
                                            max_window_size=deficient_pixel_max_window_size,
                                            method=deficient_pixel_method)
        data = remove_stray_light_task(data, stray_light_path)
        data = correct_psf_task(data, psf_model_path)

        # set up alignment mask
        observatory = int(data.meta["OBSCODE"].value)
        if observatory < 4:
            alignment_mask = lambda x, y: (x > 100) * (x < 1900) * (y > 250) * (y < 1900)  # noqa: E731
        else:
            alignment_mask = lambda x, y: (((x < 824) + (x > 1224)) * ((y < 824) + (y > 1224))  # noqa: E731
                                           * (x > 100) * (x < 1900) * (y > 100) * (y < 1900))
        data = align_task(data, mask=alignment_mask)

        # Repackage data with proper metadata
        product_code = data.meta["TYPECODE"].value + data.meta["OBSCODE"].value
        new_meta = NormalizedMetadata.load_template(product_code, "1")
        new_meta["DATE-OBS"] = data.meta["DATE-OBS"].value  # TODO: do this better and fill rest of meta

        output_header = new_meta.to_fits_header(data.wcs)
        for key in output_header:
            if (key in data.meta) and output_header[key] == "" and (key != "COMMENT") and (key != "HISTORY"):
                new_meta[key].value = data.meta[key].value

        data = NDCube(data=data.data, meta=new_meta, wcs=data.wcs, unit=data.unit, uncertainty=data.uncertainty)

        if output_filename is not None and i < len(output_filename) and output_filename[i] is not None:
            output_image_task(data, output_filename[i])
        output_data.append(data)
        logger.info("ending level 1 core flow")
    return output_data


@flow(validate_parameters=False)
def level05_core_flow(
    input_data: list[str] | list[NDCube],
    gain: float = 4.9,
    bias_level: float = 100,
    dark_level: float = 55.81,
    read_noise_level: float = 17,
    bitrate_signal: int = 16,
    quartic_coefficient_path: str | None = None,
    exposure_time: float = 49 * 1000,
    readout_line_time: float = 163/2148,
    reset_line_time: float = 163/2148,
    output_filename: str | None = None,
) -> list[NDCube]:
    """Core flow for level 0.5."""
    logger = get_run_logger()

    logger.info("beginning level 0.5 core flow")

    output_data = []
    for i, this_data in enumerate(input_data):
        data = load_image_task(this_data) if isinstance(this_data, str) else this_data
        data = decode_sqrt_data(data)
        data = update_initial_uncertainty_task(data,
                                               bias_level=bias_level,
                                               dark_level=dark_level,
                                               gain=gain,
                                               read_noise_level=read_noise_level,
                                               bitrate_signal=bitrate_signal,
                                               )
        data = perform_quartic_fit_task(data, quartic_coefficient_path)

        if data.meta["OBSCODE"].value == "4":
            scaling = {"gain": 4.9 * u.photon / u.DN,
                       "wavelength": 530. * u.nm,
                       "exposure": 49 * u.s,
                       "aperture": 49.57 * u.mm ** 2}
        else:
            scaling = {"gain": 4.9 * u.photon / u.DN,
                       "wavelength": 530. * u.nm,
                       "exposure": 49 * u.s,
                       "aperture": 34 * u.mm ** 2}
        data.data[:, :] = np.clip(dn_to_msb(data.data[:, :], data.wcs, **scaling), a_min=0, a_max=None)
        data.uncertainty.array[:, :] = dn_to_msb(data.uncertainty.array[:, :], data.wcs, **scaling)

        data = destreak_task(data,
                             exposure_time=exposure_time,
                             reset_line_time=reset_line_time,
                             readout_line_time=readout_line_time)

        # set up alignment mask
        observatory = int(data.meta["OBSCODE"].value)
        if observatory < 4:
            alignment_mask = lambda x, y: (x > 100) * (x < 1900) * (y > 250) * (y < 1900)  # noqa: E731
        else:
            alignment_mask = lambda x, y: (((x < 824) + (x > 1224)) * ((y < 824) + (y > 1224))  # noqa: E731
                                           * (x > 100) * (x < 1900) * (y > 100) * (y < 1900))

        data = align_task(data, mask=alignment_mask)

        # Repackage data with proper metadata
        product_code = data.meta["TYPECODE"].value + data.meta["OBSCODE"].value
        new_meta = NormalizedMetadata.load_template(product_code, "1")
        new_meta["DATE-OBS"] = data.meta["DATE-OBS"].value  # TODO: do this better and fill rest of meta

        output_header = new_meta.to_fits_header(data.wcs)
        for key in output_header:
            if (key in data.meta) and output_header[key] == "" and (key != "COMMENT") and (key != "HISTORY"):
                new_meta[key].value = data.meta[key].value

        data = NDCube(data=data.data, meta=new_meta, wcs=data.wcs, unit=data.unit, uncertainty=data.uncertainty)

        if output_filename is not None and i < len(output_filename) and output_filename[i] is not None:
            output_image_task(data, output_filename[i])
        output_data.append(data)
        logger.info("ending level 0.5 core flow")
    return output_data
