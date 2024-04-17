from typing import List, Union, Optional

from prefect import flow, get_run_logger

from punchbowl.data import PUNCHData
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
def level1_core_flow(
    input_data: Union[str, PUNCHData],
    quartic_coefficient_path: Optional[str] = None,
    vignetting_function_path: Optional[str] = None,
    stray_light_path: Optional[str] = None,
    deficient_pixel_map_path: Optional[str] = None,
    output_filename: Optional[str] = None,
) -> List[PUNCHData]:
    """Core flow for level 1

    Parameters
    ----------
    input_data : str
        path to the image coming into level 1 core flow
    output_filename : str
        path of where to write the output from the level 1 core flow

    Returns
    -------
    None
    """
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
    data = align_task(data)
    data = correct_psf_task(data)
    logger.info("ending level 1 core flow")

    if output_filename is not None:
        output_image_task(data, output_filename)
    return [data]
