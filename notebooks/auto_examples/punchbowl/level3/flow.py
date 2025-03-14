from datetime import datetime

from ndcube import NDCube
from prefect import flow, get_run_logger

from punchbowl.data import NormalizedMetadata
from punchbowl.data.meta import set_spacecraft_location_to_earth
from punchbowl.level3.f_corona_model import subtract_f_corona_background_task
from punchbowl.level3.low_noise import create_low_noise_task
from punchbowl.level3.polarization import convert_polarization
from punchbowl.level3.stellar import subtract_starfield_background_task
from punchbowl.util import load_image_task, output_image_task


@flow(validate_parameters=False)
def level3_PIM_flow(data_list: list[str] | list[NDCube],  # noqa: N802
                     before_f_corona_model_path: str,
                     after_f_corona_model_path: str,
                     output_filename: str | None = None,
                    reference_time: datetime | None = None) -> list[NDCube]:  # noqa: ARG001
    """Level 3 PIM flow."""
    logger = get_run_logger()

    logger.info("beginning level 3 PIM flow")
    data_list = [load_image_task(d) if isinstance(d, str) else d for d in data_list]
    data_list = [subtract_f_corona_background_task(d,
                                                   before_f_corona_model_path,
                                                   after_f_corona_model_path) for d in data_list]
    output_meta = NormalizedMetadata.load_template("PIM", "3")

    out_list = [NDCube(data=d.data, wcs=d.wcs, meta=output_meta) for d in data_list]
    for o, d in zip(out_list, data_list, strict=True):
        o.meta["DATE"] = datetime.now().isoformat()
        o.meta["DATE-AVG"] = d.meta["DATE-AVG"].value
        o.meta["DATE-OBS"] = d.meta["DATE-OBS"].value
        o.meta["DATE-BEG"] = d.meta["DATE-BEG"].value
        o.meta["DATE-END"] = d.meta["DATE-END"].value
        o = set_spacecraft_location_to_earth(o)   # noqa: PLW2901

    logger.info("ending level 3 PIM flow")

    if output_filename is not None:
        output_image_task(out_list[0], output_filename)

    return out_list


@flow(validate_parameters=False)
def level3_core_flow(data_list: list[str] | list[NDCube],
                     before_f_corona_model_path: str,
                     after_f_corona_model_path: str,
                     starfield_background_path: str | None,
                     output_filename: str | None = None) -> list[NDCube]:
    """Level 3 core flow."""
    logger = get_run_logger()

    logger.info("beginning level 3 core flow")
    data_list = [load_image_task(d) if isinstance(d, str) else d for d in data_list]
    data_list = [subtract_f_corona_background_task(d,
                                                   before_f_corona_model_path,
                                                   after_f_corona_model_path) for d in data_list]
    data_list = [subtract_starfield_background_task(d, starfield_background_path) for d in data_list]
    data_list = [convert_polarization(d) for d in data_list]
    logger.info("ending level 3 core flow")


    if output_filename is not None:
        output_image_task(data_list[0], output_filename)

    return data_list


@flow(validate_parameters=False)
def generate_level3_low_noise_flow(data_list: list[str] | list[NDCube],
                                   output_filename: str | None = None) -> list[NDCube]:
    """Generate low noise products."""
    logger = get_run_logger()

    logger.info("Generating low noise products")
    data_list = [load_image_task(d) if isinstance(d, str) else d for d in data_list]
    low_noise_image = create_low_noise_task(data_list)

    if output_filename is not None:
        output_image_task(low_noise_image, output_filename)

    return [low_noise_image]
