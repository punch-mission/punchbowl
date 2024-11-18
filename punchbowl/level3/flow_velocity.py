import os

from ndcube import NDCube
from prefect import flow, get_run_logger

from punchbowl.level3.velocity import plot_flow_map, track_velocity
from punchbowl.util import output_image_task


@flow(validate_parameters=False)
def generate_level3_velocity_flow(data_list: list[str],
                                  output_filename: str | None = None) -> list[NDCube]:
    """Generate Level 3 velocity data product."""
    logger = get_run_logger()

    logger.info("Generating velocity data product")
    velocity_data, plot_parameters = track_velocity(data_list)

    if output_filename is not None:
        output_image_task(velocity_data, output_filename)
        plot_filename = f"{os.path.splitext(output_filename)[0]}.{"png"}"
        plot_flow_map(plot_filename, **plot_parameters)

    return [velocity_data]
