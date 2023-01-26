from prefect import flow, get_run_logger

from punchbowl.level2.polarization import resolve_polarization_task
from punchbowl.level2.quality import quality_flag_task
from punchbowl.util import output_image_task, load_image_task


@flow
def level2_core_flow(input_filename, output_filename):
    logger = get_run_logger()

    logger.info("beginning level 2 core flow")
    data = load_image_task(input_filename)
    data = resolve_polarization_task(data)
    data = quality_flag_task(data)
    logger.info("ending level 2 core flow")
    output_image_task(data, output_filename)
