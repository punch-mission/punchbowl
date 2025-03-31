from datetime import UTC, datetime

import numpy as np
from astropy.nddata import StdDevUncertainty
from ndcube import NDCube
from prefect import flow, get_run_logger

from punchbowl.data import NormalizedMetadata, get_base_file_name
from punchbowl.data.meta import set_spacecraft_location_to_earth
from punchbowl.data.wcs import load_quickpunch_mosaic_wcs, load_quickpunch_nfi_wcs
from punchbowl.level2.merge import merge_many_clear_task
from punchbowl.level2.resample import reproject_many_flow
from punchbowl.util import average_datetime, load_image_task, output_image_task

ORDER_QP = ["CR1", "CR2", "CR3", "CR4"]

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

        output_data_mosaic = merge_many_clear_task(data_list_mosaic, quickpunch_mosaic_wcs, level="Q")

        output_data_mosaic.meta["DATE"] = datetime.now(UTC).isoformat()
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
        output_data_nfi.meta["DATE"] = datetime.now(UTC).isoformat()
        output_data_nfi.meta["DATE-AVG"] = output_dateobs_nfi
        output_data_nfi.meta["DATE-OBS"] = output_dateobs_nfi
        output_data_nfi.meta["DATE-BEG"] = output_datebeg_nfi
        output_data_nfi.meta["DATE-END"] = output_dateend_nfi
        output_data_nfi = set_spacecraft_location_to_earth(output_data_nfi)
    else:
        output_dateobs = datetime.now(UTC).isoformat()
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
        output_data_mosaic.meta["DATE"] = datetime.now(UTC).isoformat()
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

        output_data_nfi.meta["DATE"] = datetime.now(UTC).isoformat()
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
