from glob import glob
from datetime import datetime

import numpy as np
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube
from prefect import flow, get_run_logger

from punchbowl.data import get_base_file_name, load_trefoil_wcs
from punchbowl.data.meta import NormalizedMetadata, set_spacecraft_location_to_earth
from punchbowl.data.wcs import load_quickpunch_mosaic_wcs, load_quickpunch_nfi_wcs
from punchbowl.level2.bright_structure import identify_bright_structures_task
from punchbowl.level2.merge import merge_many_clear_task, merge_many_polarized_task
from punchbowl.level2.polarization import resolve_polarization_task
from punchbowl.level2.resample import reproject_many_flow
from punchbowl.util import average_datetime, load_image_task, output_image_task

ORDER = ["PM1", "PZ1", "PP1",
         "PM2", "PZ2", "PP2",
         "PM3", "PZ3", "PP3",
         "PM4", "PZ4", "PP4"]

ORDER_QP = ["CR1", "CR2", "CR3", "CR4"]


@flow(validate_parameters=False)
def level2_core_flow(data_list: list[str] | list[NDCube],
                     voter_filenames: list[list[str]],
                     trefoil_wcs: WCS | None = None,
                     trefoil_shape: tuple[int, int] | None = None,
                     output_filename: str | None = None) -> list[NDCube]:
    """Level 2 core flow."""
    logger = get_run_logger()
    logger.info("beginning level 2 core flow")

    data_list = [load_image_task(d) if isinstance(d, str) else d for d in data_list]

    if data_list:
        # order the data list so it can be processed properly
        output_dateobs = average_datetime([d.meta.datetime for d in data_list]).isoformat()
        output_datebeg = min([d.meta.datetime for d in data_list]).isoformat()
        output_dateend = max([d.meta.datetime for d in data_list]).isoformat()

        ordered_data_list: list[NDCube | None] = [None for _ in range(len(ORDER))]
        ordered_voters: list[list[str]] = [[] for _ in range(len(ORDER))]
        for i, order_element in enumerate(ORDER):
            for j, data_element in enumerate(data_list):
                typecode = data_element.meta["TYPECODE"].value
                obscode = data_element.meta["OBSCODE"].value
                if typecode == order_element[:2] and obscode == order_element[2]:
                    ordered_data_list[i] = data_element
                    ordered_voters[i] = voter_filenames[j]
        logger.info("Ordered files are "
                    f"{[get_base_file_name(cube) if cube is not None else None for cube in ordered_data_list]}")

        if trefoil_wcs is None or trefoil_shape is None:
            trefoil_wcs, trefoil_shape = load_trefoil_wcs()

        data_list = [resolve_polarization_task.submit(ordered_data_list[i:i+3]) for i in range(0, len(ORDER), 3)]
        data_list = [entry.result() for entry in data_list]
        data_list = reproject_many_flow([j for i in data_list for j in i], trefoil_wcs, trefoil_shape)
        data_list = [identify_bright_structures_task(cube, this_voter_filenames)
                     for cube, this_voter_filenames in zip(data_list, ordered_voters, strict=True)]
        output_data = merge_many_polarized_task(data_list, trefoil_wcs)
    else:
        output_dateobs = datetime.now().isoformat()
        output_datebeg = output_dateobs
        output_dateend = output_datebeg

        output_data = NDCube(
        data=np.zeros(trefoil_shape),
        uncertainty=StdDevUncertainty(np.zeros(trefoil_shape)),
        wcs=trefoil_wcs,
        meta=NormalizedMetadata.load_template("PTM", "2"),
    )

    output_data.meta["DATE"] = datetime.now().isoformat()
    output_data.meta["DATE-AVG"] = output_dateobs
    output_data.meta["DATE-OBS"] = output_dateobs
    output_data.meta["DATE-BEG"] = output_datebeg
    output_data.meta["DATE-END"] = output_dateend
    output_data = set_spacecraft_location_to_earth(output_data)

    if output_filename is not None:
        output_image_task(output_data, output_filename)

    logger.info("ending level 2 core flow")
    return [output_data]


# TODO: add bright structure id?
@flow(validate_parameters=False)
def levelq_core_flow(data_list: list[str] | list[NDCube],
                     output_filename: list[str] | None = None) -> list[NDCube]:
    """Level quickPUNCH core flow."""
    logger = get_run_logger()
    logger.info("beginning level quickPUNCH core flow")

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
        quickpunch_nfi_wcs, quickpunch_nfi_shape = load_quickpunch_nfi_wcs()

        data_list_mosaic = reproject_many_flow(ordered_data_list, quickpunch_mosaic_wcs, quickpunch_mosaic_shape)
        output_dateobs = average_datetime([d.meta.datetime for d in data_list_mosaic]).isoformat()
        output_datebeg = min([d.meta.datetime for d in data_list_mosaic]).isoformat()
        output_dateend = max([d.meta.datetime for d in data_list_mosaic]).isoformat()

        data_list_nfi = reproject_many_flow(ordered_data_list[-1:], quickpunch_nfi_wcs, quickpunch_nfi_shape)
        output_dateobs_nfi = average_datetime([d.meta.datetime for d in data_list_nfi]).isoformat()
        output_datebeg_nfi = min([d.meta.datetime for d in data_list_nfi]).isoformat()
        output_dateend_nfi = max([d.meta.datetime for d in data_list_nfi]).isoformat()

        output_data_mosaic = merge_many_clear_task(data_list_mosaic, quickpunch_mosaic_wcs)

        output_data_mosaic.meta["DATE"] = datetime.now().isoformat()
        output_data_mosaic.meta["DATE-AVG"] = output_dateobs
        output_data_mosaic.meta["DATE-OBS"] = output_dateobs
        output_data_mosaic.meta["DATE-BEG"] = output_datebeg
        output_data_mosaic.meta["DATE-END"] = output_dateend
        output_data_mosaic = set_spacecraft_location_to_earth(output_data_mosaic)

        output_meta_nfi = NormalizedMetadata.load_template("CNN", "Q")
        output_meta_nfi["DATE-OBS"] = data_list_nfi[0].meta["DATE-OBS"].value
        output_data_nfi =  NDCube(
            data=data_list_nfi[0].data,
            uncertainty=StdDevUncertainty(data_list_nfi[0].uncertainty.array),
            wcs=quickpunch_nfi_wcs,
            meta=output_meta_nfi,
            )
        output_data_nfi.meta["DATE"] = datetime.now().isoformat()
        output_data_nfi.meta["DATE-AVG"] = output_dateobs_nfi
        output_data_nfi.meta["DATE-OBS"] = output_dateobs_nfi
        output_data_nfi.meta["DATE-BEG"] = output_datebeg_nfi
        output_data_nfi.meta["DATE-END"] = output_dateend_nfi
        output_data_nfi = set_spacecraft_location_to_earth(output_data_nfi)
    else:
        output_dateobs = datetime.now().isoformat()
        output_datebeg = output_dateobs
        output_dateend = output_datebeg

        quickpunch_mosaic_wcs, quickpunch_mosaic_shape = load_quickpunch_mosaic_wcs()
        quickpunch_nfi_wcs, quickpunch_nfi_shape = load_quickpunch_nfi_wcs()
        output_data_mosaic = NDCube(
            data=np.zeros(quickpunch_mosaic_shape),
            uncertainty=StdDevUncertainty(np.zeros(quickpunch_mosaic_shape)),
            wcs=quickpunch_mosaic_wcs,
            meta=NormalizedMetadata.load_template("CTM", "Q"),
        )
        output_data_mosaic.meta["DATE"] = datetime.now().isoformat()
        output_data_mosaic.meta["DATE-AVG"] = output_dateobs
        output_data_mosaic.meta["DATE-OBS"] = output_dateobs
        output_data_mosaic.meta["DATE-BEG"] = output_datebeg
        output_data_mosaic.meta["DATE-END"] = output_dateend
        output_data_mosaic = set_spacecraft_location_to_earth(output_data_mosaic)

        output_data_nfi = NDCube(
            data=np.zeros(quickpunch_nfi_shape),
            uncertainty=StdDevUncertainty(np.zeros(quickpunch_nfi_shape)),
            wcs=quickpunch_nfi_wcs,
            meta=NormalizedMetadata.load_template("CNN", "Q"),
        )

        output_data_nfi.meta["DATE"] = datetime.now().isoformat()
        output_data_nfi.meta["DATE-AVG"] = output_dateobs
        output_data_nfi.meta["DATE-OBS"] = output_dateobs
        output_data_nfi.meta["DATE-BEG"] = output_dateobs
        output_data_nfi.meta["DATE-END"] = output_dateobs
        output_data_nfi = set_spacecraft_location_to_earth(output_data_nfi)


    if output_filename is not None:
        output_image_task(output_data_mosaic, output_filename[0])
        output_image_task(output_data_nfi, output_filename[1])

    logger.info("ending level quickPUNCH core flow")
    return [output_data_mosaic, output_data_nfi]


if __name__ == "__main__":
    filenames = glob("/Users/jhughes/new_results/nov25-1026/cr/*.fits")
    out = levelq_core_flow(filenames)
