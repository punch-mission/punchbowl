
from ndcube import NDCube
from prefect import flow, get_run_logger

from punchbowl.level3.f_corona_model import subtract_f_corona_background_task
from punchbowl.level3.polarization import convert_polarization
from punchbowl.level3.stellar import subtract_starfield_background_task
from punchbowl.util import load_image_task, output_image_task


@flow(validate_parameters=False)
def level3_core_flow(data_list: list[str] | list[NDCube],
                     f_corona_model_path: str | None,
                     starfield_background_path: str | None,
                     output_filename: str | None = None) -> list[NDCube]:
    """Level 3 core flow."""
    logger = get_run_logger()

    logger.info("beginning level 3 core flow")
    data_list = [load_image_task(d) if isinstance(d, str) else d for d in data_list]
    data_list = [subtract_f_corona_background_task(d, f_corona_model_path) for d in data_list]
    data_list = [subtract_starfield_background_task(d, starfield_background_path) for d in data_list]
    data_list = [convert_polarization(d) for d in data_list]
    # TODO: build low noise products
    logger.info("ending level 3 core flow")

    if output_filename is not None:
        output_image_task(data_list[0], output_filename)

    return data_list
