from prefect import flow, get_run_logger, task

from punchbowl.level1.alignment import align_task
from punchbowl.level1.quartic_fit import perform_quartic_fit_task
from punchbowl.level1.despike import despike_task
from punchbowl.level1.destreak import destreak_task
from punchbowl.level1.vignette import correct_vignetting_task
from punchbowl.level1.deficient_pixel import remove_deficient_pixels_task
from punchbowl.level1.stray_light import remove_stray_light_task
from punchbowl.level1.psf import correct_psf_task
from punchbowl.level1.flagging import flag_task
from punchbowl.util import load_image_task, output_image_task


@flow
def level1_core_flow(input_filename: str, output_filename: str) -> None:
    """Core flow for level 1

    Parameters
    ----------
    input_filename : str
        path to the image coming into level 1 core flow
    output_filename : str
        path of where to write the output from the level 1 core flow

    Returns
    -------
    None
    """
    logger = get_run_logger()

    logger.info("beginning level 1 core flow")
    data = load_image_task(input_filename)
    data = perform_quartic_fit_task(data)
    data = despike_task(data)
    data = destreak_task(data)
    data = correct_vignetting_task(data)
    data = remove_deficient_pixels_task(data)
    data = remove_stray_light_task(data)
    data = align_task(data)
    data = correct_psf_task(data)
    data = flag_task(data)
    logger.info("ending level 1 core flow")
    output_image_task(data, output_filename)
