from ndcube import NDCube
from prefect import flow, get_run_logger

from punchbowl.level3.velocity import track_velocity_task
from punchbowl.util import load_image_task, output_image_task


@flow(validate_parameters=False)
def generate_level3_velocity_flow(data_list: list[str] | list[NDCube],
                                  output_filename: str | None = None) -> list[NDCube]:
    """Generate Level 3 velocity data product."""
    logger = get_run_logger()

    logger.info("Generating velocity data product")
    data_list = [load_image_task(d) if isinstance(d, str) else d for d in data_list]
    velocity_data = track_velocity_task(data_list)

    if output_filename is not None:
        output_image_task(velocity_data, output_filename)

    return [velocity_data]
