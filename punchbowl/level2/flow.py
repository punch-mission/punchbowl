from prefect import flow, get_run_logger

from punchbowl.level2.polarization import resolve_polarization_task
from punchbowl.level2.quality import quality_flag_task
from punchbowl.level2.bright_structure import identify_bright_structures_task
from punchbowl.level2.image_merge import image_merge_flow
from punchbowl.util import load_image_task, output_image_task


@flow
def level2_core_flow(input_filename, output_filename):
    logger = get_run_logger()

    logger.info("beginning level 2 core flow")
    data = load_image_task(input_filename)
    data = resolve_polarization_task(data)
    data = identify_bright_structures_task(data)
    data = quality_flag_task(data)
    data = image_merge_flow(data)
    logger.info("ending level 2 core flow")
    output_image_task(data, output_filename)
