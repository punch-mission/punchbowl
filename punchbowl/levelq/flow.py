from datetime import UTC, datetime
from collections.abc import Callable

import numpy as np
from astropy.nddata import StdDevUncertainty
from ndcube import NDCube
from prefect import flow, get_run_logger

from punchbowl.data import NormalizedMetadata, get_base_file_name
from punchbowl.data.meta import set_spacecraft_location_to_earth
from punchbowl.data.wcs import load_quickpunch_mosaic_wcs, load_quickpunch_nfi_wcs
from punchbowl.level2.merge import merge_many_clear_task
from punchbowl.level2.resample import reproject_many_flow
from punchbowl.levelq.pca import pca_filter
from punchbowl.util import average_datetime, load_image_task, output_image_task

ORDER_QP = ["CR1", "CR2", "CR3", "CNN"]

@flow(validate_parameters=False)
def levelq_CNN_core_flow(data_list: list[str] | list[NDCube], #noqa: N802
                         output_filename: list[str] | None = None,
                         files_to_fit: list[str | NDCube | Callable] | None = None) -> list[NDCube]:
    """Level quickPUNCH NFI core flow."""
    logger = get_run_logger()
    logger.info("beginning level quickPUNCH CNN core flow")

    output_cubes = []
    for i, input_file in enumerate(data_list):
        data_cube = load_image_task(input_file) if isinstance(input_file, str) else input_file
        if files_to_fit:
            pca_filter(data_cube, files_to_fit)

        quickpunch_nfi_wcs, quickpunch_nfi_shape = load_quickpunch_nfi_wcs()

        data_list_nfi = reproject_many_flow([data_cube], quickpunch_nfi_wcs, quickpunch_nfi_shape)
        data = data_list_nfi[0].data
        uncertainty = data_list_nfi[0].uncertainty.array

        isnan = np.isnan(data)
        uncertainty[isnan] = np.inf
        data[isnan] = 0

        output_meta_nfi = NormalizedMetadata.load_template("CNN", "Q")
        output_cube = NDCube(
            data=data,
            uncertainty=StdDevUncertainty(uncertainty),
            wcs=quickpunch_nfi_wcs,
            meta=output_meta_nfi,
            )
        output_cube.meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        output_cube.meta["DATE-AVG"] = data_cube.meta["DATE-AVG"].value
        output_cube.meta["DATE-OBS"] = data_cube.meta["DATE-OBS"].value
        output_cube.meta["DATE-BEG"] = data_cube.meta["DATE-BEG"].value
        output_cube.meta["DATE-END"] = data_cube.meta["DATE-END"].value
        output_cube.meta["FILEVRSN"] = data_cube.meta["FILEVRSN"].value
        set_spacecraft_location_to_earth(output_cube)

        output_cubes.append(output_cube)

        if output_filename is not None and i < len(output_filename) and output_filename[i] is not None:
            output_image_task(output_cube, output_filename[i])

    logger.info("ending level quickPUNCH CNN core flow")
    return output_cubes


@flow(validate_parameters=False)
def levelq_CTM_core_flow(data_list: list[str] | list[NDCube], #noqa: N802
                     output_filename: list[str] | None = None) -> list[NDCube]:
    """Level quickPUNCH core flow."""
    logger = get_run_logger()
    logger.info("beginning level quickPUNCH CTM core flow")

    data_list = [load_image_task(d) if isinstance(d, str) else d for d in data_list]

    if data_list:
        ordered_data_list: list[NDCube | None] = [None for _ in range(len(ORDER_QP))]
        for i, order_element in enumerate(ORDER_QP):
            for data_element in data_list:
                typecode = data_element.meta["TYPECODE"].value
                obscode = data_element.meta["OBSCODE"].value
                if typecode == order_element[:2] and obscode == order_element[2]:
                    ordered_data_list[i] = data_element
        logger.info("Ordered files are "
                    f"{[get_base_file_name(cube) if cube is not None else None for cube in ordered_data_list]}")

        quickpunch_mosaic_wcs, quickpunch_mosaic_shape = load_quickpunch_mosaic_wcs()

        data_list_mosaic = reproject_many_flow(ordered_data_list, quickpunch_mosaic_wcs, quickpunch_mosaic_shape)
        output_dateobs = average_datetime(
            [d.meta.datetime for d in data_list_mosaic if d is not None],
        ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        output_datebeg = min([d.meta.datetime for d in data_list_mosaic if d is not None],
                             ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        output_dateend = max([d.meta.datetime for d in data_list_mosaic if d is not None],
                             ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

        output_data_mosaic = merge_many_clear_task(data_list_mosaic, quickpunch_mosaic_wcs, level="Q")

        output_data_mosaic.meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        output_data_mosaic.meta["DATE-AVG"] = output_dateobs
        output_data_mosaic.meta["DATE-OBS"] = output_dateobs
        output_data_mosaic.meta["DATE-BEG"] = output_datebeg
        output_data_mosaic.meta["DATE-END"] = output_dateend
        output_data_mosaic.meta["FILEVRSN"] = ordered_data_list[0].meta["FILEVRSN"].value
        output_data_mosaic = set_spacecraft_location_to_earth(output_data_mosaic)
    else:
        output_dateobs = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        output_datebeg = output_dateobs
        output_dateend = output_datebeg

        quickpunch_mosaic_wcs, quickpunch_mosaic_shape = load_quickpunch_mosaic_wcs()
        output_data_mosaic = NDCube(
            data=np.zeros(quickpunch_mosaic_shape),
            uncertainty=StdDevUncertainty(np.zeros(quickpunch_mosaic_shape)),
            wcs=quickpunch_mosaic_wcs,
            meta=NormalizedMetadata.load_template("CTM", "Q"),
        )
        output_data_mosaic.meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        output_data_mosaic.meta["DATE-AVG"] = output_dateobs
        output_data_mosaic.meta["DATE-OBS"] = output_dateobs
        output_data_mosaic.meta["DATE-BEG"] = output_datebeg
        output_data_mosaic.meta["DATE-END"] = output_dateend
        output_data_mosaic = set_spacecraft_location_to_earth(output_data_mosaic)

    if output_filename is not None:
        output_image_task(output_data_mosaic, output_filename[0])

    logger.info("ending level quickPUNCH CTM core flow")
    return [output_data_mosaic]
