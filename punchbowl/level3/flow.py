import os
from datetime import UTC, datetime

from ndcube import NDCube
from prefect import get_run_logger

from punchbowl.data.meta import NormalizedMetadata, set_spacecraft_location_to_earth
from punchbowl.level3.f_corona_model import subtract_f_corona_background_task
from punchbowl.level3.low_noise import create_low_noise_task
from punchbowl.level3.polarization import convert_polarization
from punchbowl.level3.stellar import subtract_starfield_background_task
from punchbowl.level3.velocity import plot_flow_map, track_velocity
from punchbowl.prefect import punch_flow
from punchbowl.util import load_image_task, output_image_task


@punch_flow
def level3_PIM_flow(data_list: list[str] | list[NDCube],  # noqa: N802
                     before_f_corona_model_path: str,
                     after_f_corona_model_path: str,
                     output_filename: str | None = None,
                     reference_time: datetime | None = None) -> list[NDCube]:  # noqa: ARG001
    """Level 3 PIM/CIM flow."""
    logger = get_run_logger()

    logger.info("beginning level 3 PIM/CIM flow")
    data_list = [load_image_task(d) if isinstance(d, str) else d for d in data_list]
    new_type = "CIM" if data_list[0].meta["TYPECODE"].value == "CT" else "PIM"

    for cube in data_list:
        # We'll want to grab the history we accumulate through this flow and put it in the final product,
        # but the per-file history up to now is kind of meaningless for the merged final product.
        if cube is not None:
            cube.meta.history.clear()

    data_list = [subtract_f_corona_background_task(d,
                                                   before_f_corona_model_path,
                                                   after_f_corona_model_path) for d in data_list]
    output_meta = NormalizedMetadata.load_template(new_type, "3")

    out_list = [NDCube(data=d.data, wcs=d.wcs, meta=output_meta) for d in data_list]
    for o, d in zip(out_list, data_list, strict=True):
        o.meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        o = set_spacecraft_location_to_earth(o)   # noqa: PLW2901
        o.meta.history = d.meta.history
        o.meta["CALFCOR1"] = os.path.basename(before_f_corona_model_path)
        o.meta["CALFCOR2"] = os.path.basename(after_f_corona_model_path)
        for key in ["FILEVRSN", "ALL_INPT", "HAS_WFI1", "HAS_WFI2", "HAS_WFI3", "HAS_NFI4", "DATE-AVG", "DATE-OBS",
                    "DATE-BEG", "DATE-END"]:
            o.meta[key] = d.meta[key].value

    logger.info("ending level 3 PIM/CIM flow")

    for o in out_list:
        o.meta.provenance = [fname for d in data_list if d is not None and (fname := d.meta.get("FILENAME").value)]

    if output_filename is not None:
        output_image_task(out_list[0], output_filename)

    return out_list


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
                    "DATE-BEG", "DATE-END"]:
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
                                   output_filename: str | None = None) -> list[NDCube]:
    """Generate low noise products."""
    logger = get_run_logger()

    logger.info("Generating low noise products")
    data_list = [load_image_task(d) if isinstance(d, str) else d for d in data_list]
    low_noise_image = create_low_noise_task(data_list)

    provenance_list = [fname for d in data_list if d is not None and (fname := d.meta.get("FILENAME").value)]
    low_noise_image.meta.provenance = provenance_list

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
