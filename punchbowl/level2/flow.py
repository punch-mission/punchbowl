from datetime import UTC, datetime

import numpy as np
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube
from prefect import get_run_logger

from punchbowl.data import get_base_file_name, load_trefoil_wcs
from punchbowl.data.meta import NormalizedMetadata, set_spacecraft_location_to_earth
from punchbowl.level2.bright_structure import identify_bright_structures_task
from punchbowl.level2.merge import merge_many_clear_task, merge_many_polarized_task
from punchbowl.level2.polarization import resolve_polarization_task
from punchbowl.level2.resample import reproject_many_flow
from punchbowl.prefect import punch_flow
from punchbowl.util import average_datetime, find_first_existing_file, load_image_task, output_image_task

POLARIZED_FILE_ORDER = ["PM1", "PZ1", "PP1",
                        "PM2", "PZ2", "PP2",
                        "PM3", "PZ3", "PP3",
                        "PM4", "PZ4", "PP4"]


@punch_flow
def level2_core_flow(data_list: list[str] | list[NDCube],
                     voter_filenames: list[list[str]],
                     polarized: bool | None = None,
                     trefoil_wcs: WCS | None = None,
                     trefoil_shape: tuple[int, int] | None = None,
                     output_filename: str | None = None) -> list[NDCube]:
    """
    Level 2 core flow.

    Parameters
    ----------
    data_list : list[str] | list[NDCube]
        The files or data cubes to be merged into a mosaic
    voter_filenames : list[list[str]]
        The voter files for detecting bright structures
    polarized : bool
        Whether to generate a polarized or clear mosaic. Only required if `data_list` is not provided (and so an empty
        cube is being generated). Otherwise, this is auto-detected.
    trefoil_wcs : WCS | None
        The frame to build the mosaic in. By default, the default trefoil mosaic is used.
    trefoil_shape : tuple[int, int] | None
        The size of the frame to build the mosaic in. By default, the default trefoil size is used.
    output_filename : str | None
        If provided, the resulting mosaic is written to this path.

    Returns
    -------
    output_data: list[NDCube]
        The resulting data cube. For compatibility, it will be a list of a single cube.

    """
    logger = get_run_logger()
    logger.info("beginning level 2 core flow")

    data_list = [load_image_task(d) if isinstance(d, str) else d for d in data_list]

    if data_list and not all(cube is None for cube in data_list):
        if polarized is None:
            polarized = data_list[0].meta["TYPECODE"].value[0] == "P"

        output_dateobs = average_datetime([d.meta.datetime for d in data_list]).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        output_datebeg = min([d.meta.datetime for d in data_list]).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        output_dateend = max([d.meta.datetime for d in data_list]).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

        if polarized:
            # order the data list so it can be processed properly
            ordered_data_list: list[NDCube | None] = [None for _ in range(len(POLARIZED_FILE_ORDER))]
            ordered_voters: list[list[str]] = [[] for _ in range(len(POLARIZED_FILE_ORDER))]
            for i, order_element in enumerate(POLARIZED_FILE_ORDER):
                for j, data_element in enumerate(data_list):
                    typecode = data_element.meta["TYPECODE"].value
                    obscode = data_element.meta["OBSCODE"].value
                    if typecode == order_element[:2] and obscode == order_element[2]:
                        ordered_data_list[i] = data_element
                        ordered_voters[i] = voter_filenames[j]
            logger.info("Ordered files are "
                        f"{[get_base_file_name(cube) if cube is not None else None for cube in ordered_data_list]}")
            data_list = [resolve_polarization_task.submit(ordered_data_list[i:i+3])
                         for i in range(0, len(POLARIZED_FILE_ORDER), 3)]
            data_list = [entry.result() for entry in data_list]
            data_list = [j for i in data_list for j in i]
            voter_filenames = ordered_voters

        default_trefoil_wcs, default_trefoil_shape = load_trefoil_wcs()
        trefoil_wcs = trefoil_wcs or default_trefoil_wcs
        trefoil_shape = trefoil_shape or default_trefoil_shape

        data_list = reproject_many_flow(data_list, trefoil_wcs, trefoil_shape)
        data_list = [identify_bright_structures_task(cube, this_voter_filenames)
                     for cube, this_voter_filenames in zip(data_list, voter_filenames, strict=True)]
        merger = merge_many_polarized_task if polarized else merge_many_clear_task
        output_data = merger(data_list, trefoil_wcs)
        output_data.meta["FILEVRSN"] = find_first_existing_file(data_list).meta["FILEVRSN"].value
    else:
        if polarized is None:
            msg = "A polarization state must be provided"
            raise ValueError(msg)
        output_dateobs = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        output_datebeg = output_dateobs
        output_dateend = output_datebeg

        output_data = NDCube(
            data=np.zeros(trefoil_shape),
            uncertainty=StdDevUncertainty(np.zeros(trefoil_shape)),
            wcs=trefoil_wcs,
            meta=NormalizedMetadata.load_template("PTM" if polarized else "CTM", "2"),
        )

    output_data.meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    output_data.meta["DATE-AVG"] = output_dateobs
    output_data.meta["DATE-OBS"] = output_dateobs
    output_data.meta["DATE-BEG"] = output_datebeg
    output_data.meta["DATE-END"] = output_dateend
    output_data = set_spacecraft_location_to_earth(output_data)

    if output_filename is not None:
        output_image_task(output_data, output_filename)

    logger.info("ending level 2 core flow")
    return [output_data]
