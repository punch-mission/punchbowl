import os
from datetime import UTC, datetime

import numpy as np
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube
from prefect import flow, get_run_logger

from punchbowl.data import NormalizedMetadata, get_base_file_name
from punchbowl.data.meta import set_spacecraft_location_to_earth
from punchbowl.data.punch_io import load_many_cubes
from punchbowl.data.wcs import load_quickpunch_mosaic_wcs, load_quickpunch_nfi_wcs
from punchbowl.level2.merge import merge_many_clear_task
from punchbowl.level2.preprocess import preprocess_trefoil_inputs
from punchbowl.level2.resample import find_central_pixel, reproject_many_flow
from punchbowl.levelq.pca import pca_filter
from punchbowl.util import DataLoader, average_datetime, find_first_existing_file, load_image_task, output_image_task

ORDER_QP = ["QR1", "QR2", "QR3", "CNN"]

SPACECRAFT_OBSCODE = {"1": "WFI1",
                      "2": "WFI2",
                      "3": "WFI3",
                      "4": "NFI4",
                      "N": "NFI4"}

@flow(validate_parameters=False)
def levelq_CNN_core_flow(data_list: list[str] | list[NDCube], #noqa: N802
                         output_filename: list[str] | None = None,
                         files_to_fit: list[str | NDCube | DataLoader] | None = None,
                         data_root: str | None = None) -> list[NDCube]:
    """
    Run the LQ CNN flow.

    This flow is designed to run on a batch of input CR4 images to facilitate more efficient PCA fitting.

    Parameters
    ----------
    data_list : list[str | NDCube]
        The input images, either as paths or NDCubes
    output_filename : list[str]
        Optional output paths at which the CNN files should be written
    files_to_fit : list[str | NDCube | DataLoader]
        Additional files to use for the PCA fitting, but not to actually be filtered or output
    data_root : str
        The root directory which the paths in ``data_list`` are relative to

    Returns
    -------
    output_cubes : list[NDCube]
        The CNN data cubes

    """
    logger = get_run_logger()
    logger.info("beginning level quickPUNCH CNN core flow")
    logger.info(f"Got {len(data_list)} input files and {len(files_to_fit)} extra files for fitting")

    output_cubes = []

    data_cubes = [input_file for input_file in data_list if isinstance(input_file, NDCube)]
    input_paths = [input_file for input_file in data_list if isinstance(input_file, str)]
    if data_root is not None:
        input_paths = [os.path.join(data_root, path) for path in input_paths]

    data_cubes += load_many_cubes(input_paths, n_workers=3)

    logger.info("Loaded images to be subtracted")

    pca_filter(data_cubes, files_to_fit)

    quickpunch_nfi_wcs, quickpunch_nfi_shape = load_quickpunch_nfi_wcs()
    data_list_nfi = reproject_many_flow(data_cubes, quickpunch_nfi_wcs, quickpunch_nfi_shape)

    for i, data_cube in enumerate(data_list_nfi):
        data = data_cube.data
        uncertainty = data_cube.uncertainty.array

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
def levelq_CTM_core_flow(data_list: list[str] | list[NDCube], #noqa: N802, C901
                         output_filename: list[str] | None = None,
                         trim_edges_px: int = 0,
                         alphas_file: str | None = None,
                         trefoil_wcs: WCS | None = None,
                         trefoil_shape: tuple[int, int] | None = None,
                         ) -> list[NDCube]:
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
        if trefoil_wcs is not None:
            quickpunch_mosaic_wcs = trefoil_wcs
        if trefoil_shape is not None:
            quickpunch_mosaic_shape = trefoil_shape

        preprocess_trefoil_inputs(data_list, trim_edges_px, alphas_file)

        data_list_mosaic = reproject_many_flow(ordered_data_list, quickpunch_mosaic_wcs, quickpunch_mosaic_shape)
        output_dateobs = average_datetime(
            [d.meta.datetime for d in data_list_mosaic if d is not None],
        ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        output_datebeg = min([d.meta.datetime for d in data_list_mosaic if d is not None],
                             ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        output_dateend = max([d.meta.datetime for d in data_list_mosaic if d is not None],
                             ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

        output_data_mosaic = merge_many_clear_task(data_list_mosaic, quickpunch_mosaic_wcs, level="Q")

        for d in filter(None, data_list_mosaic):
            spacecraft = SPACECRAFT_OBSCODE[d.meta["OBSCODE"].value]
            output_data_mosaic.meta[f"HAS_{spacecraft}"] = 1

        output_data_mosaic.meta["ALL_INPT"] = {output_data_mosaic.meta["HAS_WFI1"],
                                               output_data_mosaic.meta["HAS_WFI2"],
                                               output_data_mosaic.meta["HAS_WFI3"],
                                               output_data_mosaic.meta["HAS_NFI4"]} == {1}

        centers = find_central_pixel(ordered_data_list, quickpunch_mosaic_wcs)
        for center, cube in zip(centers, ordered_data_list, strict=False):
            if center is None:
                continue
            cx, cy = center
            obs_no = cube.meta["OBSCODE"].value
            obs = "NFI" if obs_no == "N" else "WFI"
            obs_no = "4" if obs_no == "N" else obs_no
            output_data_mosaic.meta[f"CTRX{obs}{obs_no}"] = cx
            output_data_mosaic.meta[f"CTRY{obs}{obs_no}"] = cy

        output_data_mosaic.meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        output_data_mosaic.meta["DATE-AVG"] = output_dateobs
        output_data_mosaic.meta["DATE-OBS"] = output_dateobs
        output_data_mosaic.meta["DATE-BEG"] = output_datebeg
        output_data_mosaic.meta["DATE-END"] = output_dateend
        output_data_mosaic.meta["FILEVRSN"] = find_first_existing_file(data_list).meta["FILEVRSN"].value
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
