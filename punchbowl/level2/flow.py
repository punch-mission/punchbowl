from astropy.nddata import StdDevUncertainty
from ndcube import NDCube
from prefect import flow, get_run_logger

from punchbowl.data import get_base_file_name, load_trefoil_wcs
from punchbowl.data.meta import NormalizedMetadata
from punchbowl.data.wcs import load_quickpunch_mosaic_wcs, load_quickpunch_nfi_wcs
from punchbowl.exceptions import IncorrectFileCountError
from punchbowl.level2.bright_structure import identify_bright_structures_task
from punchbowl.level2.merge import merge_many_clear_task, merge_many_polarized_task
from punchbowl.level2.polarization import resolve_polarization_task
from punchbowl.level2.resample import reproject_many_flow
from punchbowl.util import load_image_task, output_image_task

ORDER = ["PM1", "PZ1", "PP1",
         "PM2", "PZ2", "PP2",
         "PM3", "PZ3", "PP3",
         "PM4", "PZ4", "PP4"]

ORDER_QP = ["CR1", "CR2", "CR3", "CR4"]


@flow(validate_parameters=False)
def level2_core_flow(data_list: list[str] | list[NDCube],
                     voter_filenames: list[list[str]],
                     output_filename: str | None = None) -> list[NDCube]:
    """Level 2 core flow."""
    logger = get_run_logger()
    logger.info("beginning level 2 core flow")
    if len(data_list) != 12:
        msg = f"Received {len(data_list)} files when need 12."
        raise IncorrectFileCountError(msg)

    data_list = [load_image_task(d) if isinstance(d, str) else d for d in data_list]

    # order the data list so it can be processed properly
    ordered_data_list = []
    for order_element in ORDER:
        found = False
        for data_element in data_list:
            typecode = data_element.meta["TYPECODE"].value
            obscode = data_element.meta["OBSCODE"].value
            if typecode == order_element[:2] and obscode == order_element[2]:
                ordered_data_list.append(data_element)
                found = True
        if not found:
            msg = f"Did not receive {order_element} file."
            raise IncorrectFileCountError(msg)
    logger.info(f"Ordered files are {[get_base_file_name(cube) for cube in ordered_data_list]}")

    trefoil_wcs, trefoil_shape = load_trefoil_wcs()

    # TODO make more robust... fails if a file is missing or out of order right now
    data_list = (resolve_polarization_task(ordered_data_list[:3]) + resolve_polarization_task(ordered_data_list[3:6])
                 + resolve_polarization_task(ordered_data_list[6:9]) + resolve_polarization_task(ordered_data_list[9:]))
    data_list = reproject_many_flow(data_list, trefoil_wcs, trefoil_shape)
    data_list = [identify_bright_structures_task(cube, voter_filenames)
                 for cube, voter_filenames in zip(data_list, voter_filenames, strict=True)]
    output_data = merge_many_polarized_task(data_list, trefoil_wcs)

    if output_filename is not None:
        output_image_task(output_data, output_filename)

    logger.info("ending level 2 core flow")
    return [output_data]


@flow(validate_parameters=False)
def levelq_core_flow(data_list: list[str] | list[NDCube],
                     output_filename: list[str] | None = None) -> list[NDCube]:
    """Level quickPUNCH core flow."""
    logger = get_run_logger()
    logger.info("beginning level quickPUNCH core flow")
    if len(data_list) != 4:
        msg = f"Received {len(data_list)} files when need 4."
        raise IncorrectFileCountError(msg)

    data_list = [load_image_task(d) if isinstance(d, str) else d for d in data_list]

    # order the data list so it can be processed properly
    ordered_data_list = []
    for order_element in ORDER_QP:
        found = False
        for data_element in data_list:
            typecode = data_element.meta["TYPECODE"].value
            obscode = data_element.meta["OBSCODE"].value
            if typecode == order_element[:2] and obscode == order_element[2]:
                ordered_data_list.append(data_element)
                found = True
        if not found:
            msg = f"Did not receive {order_element} file."
            raise IncorrectFileCountError(msg)
    logger.info(f"Ordered files are {[get_base_file_name(cube) for cube in ordered_data_list]}")

    quickpunch_mosaic_wcs, quickpunch_mosaic_shape = load_quickpunch_mosaic_wcs()
    quickpunch_nfi_wcs, quickpunch_nfi_shape = load_quickpunch_nfi_wcs()

    data_list_mosaic = reproject_many_flow(ordered_data_list, quickpunch_mosaic_wcs, quickpunch_mosaic_shape)
    data_list_nfi = reproject_many_flow(ordered_data_list[-1:], quickpunch_nfi_wcs, quickpunch_nfi_shape)

    output_data_mosaic = merge_many_clear_task(data_list_mosaic, quickpunch_mosaic_wcs)

    output_meta_nfi = NormalizedMetadata.load_template("CNN", "Q")
    output_meta_nfi["DATE-OBS"] = data_list_nfi[0].meta["DATE-OBS"].value
    output_data_nfi =  NDCube(
        data=data_list_nfi[0].data,
        uncertainty=StdDevUncertainty(data_list_nfi[0].uncertainty.array),
        wcs=quickpunch_nfi_wcs,
        meta=output_meta_nfi,
        )

    if output_filename is not None:
        output_image_task(output_data_mosaic, output_filename[0])
        output_image_task(output_data_nfi, output_filename[1])

    logger.info("ending level quickPUNCH core flow")
    return [output_data_mosaic, output_data_nfi]
