import os
from datetime import UTC, datetime

from ndcube import NDCube
from prefect import get_run_logger

from punchbowl.data import load_ndcube_from_fits
from punchbowl.data.meta import NormalizedMetadata, set_spacecraft_location_to_earth
from punchbowl.level2.finalize import finalize_output
from punchbowl.level2.merge import merge_many_clear_task, merge_many_polarized_task
from punchbowl.level3.f_corona_model import subtract_f_corona_background_task
from punchbowl.level3.low_noise import create_low_noise_task
from punchbowl.level3.polarization import convert_polarization
from punchbowl.level3.stellar import subtract_starfield_background_task
from punchbowl.level3.velocity import plot_flow_map, track_velocity
from punchbowl.prefect import punch_flow
from punchbowl.util import load_image_task, output_image_task


@punch_flow
def level3_PIM_CIM_flow(data_list: list[str] | list[NDCube],  # noqa: N802
                        before_f_corona_model_paths: list[str],
                        after_f_corona_model_paths: list[str],
                        output_filename: str | None = None) -> list[NDCube]:
    """Level 3 PIM/CIM flow."""
    logger = get_run_logger()

    logger.info("beginning level 3 PIM/CIM flow")
    data_list = [load_image_task(d) if isinstance(d, str) else d for d in data_list]
    polarized = data_list[0].meta["TYPECODE"].value[1] != "R"
    new_type = "PIM" if polarized else "CIM"
    trefoil_wcs = data_list[0].wcs

    before_f_corona_models = [load_ndcube_from_fits(path) for path in before_f_corona_model_paths]
    after_f_corona_models = [load_ndcube_from_fits(path) for path in after_f_corona_model_paths]

    data_list = [subtract_f_corona_background_task(d,
                                                   before_f_corona_models,
                                                   after_f_corona_models) for d in data_list]

    merger = merge_many_polarized_task if polarized else merge_many_clear_task
    output_data = merger(data_list, trefoil_wcs, level="3", product_code=new_type)

    fcor_files = [c.meta["FILENAME"].value.replace(".fits", "") for c in before_f_corona_models + after_f_corona_models]
    output_data.meta.history.add_now("LEVEL3-subtract_f_corona_background",
                                     f"subtracted f corona background using {', '.join(fcor_files)}")

    finalize_output(output_data, data_list)

    for cube in data_list:
        obs_no = int(cube.meta["OBSCODE"].value)
        obs = "NFI" if obs_no == "4" else "WFI"
        if cube.meta[f"CTRX{obs}{obs_no}"].value > 0:
            output_data[0].meta[f"CTRX{obs}{obs_no}"] = cube.meta[f"CTRX{obs}{obs_no}"].value
            output_data[0].meta[f"CTRY{obs}{obs_no}"] = cube.meta[f"CTRY{obs}{obs_no}"].value

    logger.info("ending level 3 PIM/CIM flow")

    if output_filename is not None:
        output_image_task(output_data, output_filename)

    return [output_data]


@punch_flow
def level3_core_flow(data_list: list[str] | list[NDCube],
                     starfield_background_path: str | None,
                     output_filename: str | None = None) -> list[NDCube]:
    """Level 3 CTM flow."""
    logger = get_run_logger()

    logger.info("beginning level 3 CTM flow")
    data_list = [load_image_task(d) if isinstance(d, str) else d for d in data_list]
    is_polarized = data_list[0].meta["TYPECODE"].value == "PT"
    data_list = [subtract_starfield_background_task(d, starfield_background_path) for d in data_list]
    if is_polarized:
        data_list = [convert_polarization(d) for d in data_list]

    out_data_list = []
    for o in data_list:
        out_meta: NormalizedMetadata = NormalizedMetadata.load_template("PTM" if is_polarized else "CTM", "3")
        out_meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        out_meta.provenance = [fname for d in data_list if d is not None and (fname := d.meta.get("FILENAME").value)]
        out_meta.history = o.meta.history
        out_meta["CALSTAR1"] = starfield_background_path
        for key in ["FILEVRSN", "ALL_INPT", "HAS_WFI1", "HAS_WFI2", "HAS_WFI3", "HAS_NFI4", "DATE-AVG", "DATE-OBS",
                    "DATE-BEG", "DATE-END", "CTRXWFI1", "CTRYWFI1", "CTRXWFI2", "CTRYWFI2", "CTRXWFI3", "CTRYWFI3",
                    "CTRXNFI4", "CTRYNFI4"]:
            out_meta[key] = o.meta[key].value
        output_data = NDCube(
            data=o.data,
            uncertainty=o.uncertainty,
            wcs=o.wcs,
            meta=out_meta,
        )
        output_data = set_spacecraft_location_to_earth(output_data)
        out_data_list.append(output_data)

    if output_filename is not None:
        output_image_task(out_data_list[0], output_filename)

    logger.info("ending level 3 CTM core flow")

    return out_data_list


@punch_flow
def generate_level3_low_noise_flow(data_list: list[str] | list[NDCube],
                                   output_filename: str | None = None,
                                   reference_time: str | datetime | None = None) -> list[NDCube]:
    """Generate low noise products."""
    logger = get_run_logger()

    logger.info("Generating low noise products")
    data_list = [load_image_task(d) if isinstance(d, str) else d for d in data_list]
    low_noise_image = create_low_noise_task(data_list, reference_time=reference_time)

    if output_filename is not None:
        output_image_task(low_noise_image, output_filename)

    return [low_noise_image]


@punch_flow
def generate_level3_velocity_flow(data_list: list[str],
                                  output_filename: str | None = None) -> list[NDCube]:
    """Generate Level 3 velocity data product."""
    logger = get_run_logger()

    logger.info("Generating velocity data product")
    velocity_data, plot_parameters = track_velocity(data_list)

    if output_filename is not None:
        output_image_task(velocity_data, output_filename)
        plot_filename = f"{os.path.splitext(output_filename)[0]}.png"
        plot_flow_map(plot_filename, **plot_parameters)

    return [velocity_data]
