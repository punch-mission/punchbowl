import os
import multiprocessing
from datetime import UTC, datetime

import numpy as np
from astropy.nddata import StdDevUncertainty
from ndcube import NDCube
from prefect import flow, get_run_logger

from punchbowl.data import NormalizedMetadata, get_base_file_name, load_ndcube_from_fits
from punchbowl.data.meta import set_spacecraft_location_to_earth
from punchbowl.data.wcs import load_quickpunch_mosaic_wcs, load_quickpunch_nfi_wcs
from punchbowl.level2.merge import merge_many_clear_task
from punchbowl.level2.resample import reproject_many_flow
from punchbowl.levelq.pca import pca_filter
from punchbowl.util import DataLoader, average_datetime, load_image_task, output_image_task

ORDER_QP = ["CR1", "CR2", "CR3", "CNN"]

@flow(validate_parameters=False)
def levelq_CNN_core_flow(data_list: list[str] | list[NDCube], #noqa: N802
                         output_filename: list[str] | None = None,
                         files_to_fit: list[str | NDCube | DataLoader] | None = None,
                         outlier_limits: str | None = None,
                         data_root: str | None = None) -> list[NDCube]:
    """Level quickPUNCH NFI core flow."""
    logger = get_run_logger()
    logger.info("beginning level quickPUNCH CNN core flow")
    logger.info(f"Got {len(data_list)} input files and {len(files_to_fit)} extra files for fitting")

    output_cubes = []

    data_cubes = [input_file for input_file in data_list if isinstance(input_file, NDCube)]
    input_paths = [input_file for input_file in data_list if isinstance(input_file, str)]
    if data_root is not None:
        input_paths = [os.path.join(data_root, path) for path in input_paths]

    # This parallelizes more effectively than running a lot of load_image_task in parallel, due to how Prefect would
    # schedule those tasks. Experience shows that the main thread quickly gets overwhelmed by workers send back
    # loaded images, so more than a few worker processes doesn't help anything (but this is faster than loading
    # images in series!)
    with multiprocessing.Pool(3) as p:
        data_cubes += p.map(load_ndcube_from_fits, input_paths, chunksize=10)

    if files_to_fit is not None:
        pca_filter(data_cubes, files_to_fit, outlier_limits=outlier_limits)

    quickpunch_nfi_wcs, quickpunch_nfi_shape = load_quickpunch_nfi_wcs()
    data_list_nfi = reproject_many_flow(data_cubes, quickpunch_nfi_wcs, quickpunch_nfi_shape)

    for i, data_cube in enumerate(data_cubes):
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
