from datetime import datetime

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
    # kind = list(data._cubes.keys())[0]
    #
    # # TODO: remove this ugliness
    # data[kind].meta["OBSRVTRY"] = "Z"
    # data[kind].meta["LEVEL"] = 1
    # data[kind].meta["TYPECODE"] = "ZZ"
    # data[kind].meta["VERSION"] = 1
    # data[kind].meta["SOFTVERS"] = 1
    # data[kind].meta["DATE-OBS"] = str(datetime.now())
    # data[kind].meta["DATE-AQD"] = str(datetime.now())
    # data[kind].meta["DATE-END"] = str(datetime.now())
    # data[kind].meta["POL"] = "Z"
    # data[kind].meta["STATE"] = "finished"
    # data[kind].meta["PROCFLOW"] = "?"

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
