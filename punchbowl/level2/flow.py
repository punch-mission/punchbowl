from prefect import flow, get_run_logger, task

from punchbowl.data import PUNCHData
from punchbowl.level2.polarization import resolve_polarization_task
from punchbowl.level2.quality import quality_flag_task


@task
def load_level1_task(input_filename):
    return PUNCHData.from_fits(input_filename)


@task
def output_level2_task(data, output_filename):
    return data.write(output_filename)


@flow
def level2_core_flow(input_filename, output_filename):
    logger = get_run_logger()

    logger.info("beginning level 2 core flow")
    data = load_level1_task(input_filename)
    data = resolve_polarization_task(data)
    data = quality_flag_task(data)
    logger.info("ending level 2 core flow")
    output_level2_task(data, output_filename)
