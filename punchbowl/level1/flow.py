import os

from prefect import flow, get_run_logger, task

from punchbowl.data import PUNCHData
from punchbowl.level1.alignment import align_task
from punchbowl.level1.quartic_fit import perform_quartic_fit_task
from punchbowl.level1.despike import despike_task
from punchbowl.level1.destreak import destreak_task
from punchbowl.level1.vignette import correct_vignetting_task
from punchbowl.level1.deficient_pixel import remove_deficient_pixels_task
from punchbowl.level1.stray_light import remove_stray_light_task
from punchbowl.level1.psf import correct_psf_task
from punchbowl.level1.flagging import flag_task


@task
def load_level0_task(input_filename):
    return PUNCHData.from_fits(input_filename)


@task
def output_level1_task(data, output_filename):
    output_dir = os.path.dirname(output_filename)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    return data.write(output_filename)


@flow
def level1_core_flow(input_filename, output_filename):
    logger = get_run_logger()

    logger.info("beginning level 1 core flow")
    data = load_level0_task(input_filename)
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
    output_level1_task(data, output_filename)
