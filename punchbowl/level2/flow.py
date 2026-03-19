from datetime import UTC, datetime

import numpy as np
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube
from prefect import get_run_logger

from punchbowl.data import get_base_file_name, load_trefoil_wcs
from punchbowl.data.meta import NormalizedMetadata, set_spacecraft_location_to_earth
from punchbowl.level2.bright_structure import identify_bright_structures_task
from punchbowl.level2.finalize import finalize_output
from punchbowl.level2.merge import merge_many_clear_task, merge_many_polarized_task
from punchbowl.level2.polarization import resolve_polarization_task
from punchbowl.level2.preprocess import preprocess_trefoil_inputs
from punchbowl.level2.resample import find_central_pixel, reproject_many_flow
from punchbowl.prefect import punch_flow
from punchbowl.util import find_first_existing_file, load_image_task, output_image_task

POLARIZED_FILE_ORDER = ["PM1", "PZ1", "PP1",
                        "PM2", "PZ2", "PP2",
                        "PM3", "PZ3", "PP3",
                        "PM4", "PZ4", "PP4"]

SPACECRAFT_OBSCODE = {"1": "WFI1",
                      "2": "WFI2",
                      "3": "WFI3",
                      "4": "NFI4"}


@punch_flow
def level2_core_flow(data_list: list[str] | list[NDCube], # noqa: C901
                     voter_filenames: list[list[str]],
                     polarized: bool | None = None,
                     trefoil_wcs: WCS | None = None,
                     trefoil_shape: tuple[int, int] | None = None,
                     rolloff_width: float | list[float] = .25,
                     rolloff_strength: float | list[float] = 1,
                     trim_edges_px: int | list[int] = 0,
                     alphas_file: str | None = None,
                     image_masks: list[str | None] | None = None,
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
    rolloff_width : float | list[float]
        Before reprojection, image uncertainties are enhanced at the edges, to provide a smooth rolloff in merging. This
        controls the width of that rolloff. The rolloff width will be this number, times the shortest distance from
        image-center to image-mask-edge. A list can be provided to give one value for each spacecraft.
    rolloff_strength : float | list[float]
        Before reprojection, image uncertainties are enhanced at the edges, to provide a smooth rolloff in merging. This
        controls the strength of that rolloff. Merging weights at the mask edge will be reduced by this fractional
        amount. A strength of zero means no rolloff. A list can be provided to give one value for each spacecraft.
    trim_edges_px : int | list[int]
        Before reprojection, image edges are trimmed by this amount, and the masked region is expanded by this amount. A
         list can be provided to give one value for each spacecraft.
    alphas_file : str
        File path containing alpha scalings for relative instrument scaling.
    image_masks: list[str | None] | None
        File paths containing masks to be applied before reprojection, one per input image.
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
    if image_masks is None:
        image_masks = [None] * len(data_list)

    if data_list and not all(cube is None for cube in data_list):
        for cube in data_list:
            # We'll want to grab the history we accumulate through this flow and put it in the final product,
            # but the per-file history up to now is kind of meaningless for the merged final product.
            if cube is not None:
                cube.meta.history.clear()
        if polarized is None:
            polarized = data_list[0].meta["TYPECODE"].value[0] == "P"

        if polarized:
            # order the data list so it can be processed properly
            ordered_data_list: list[NDCube | None] = [None for _ in range(len(POLARIZED_FILE_ORDER))]
            ordered_mask_list = [None for _ in range(len(POLARIZED_FILE_ORDER))]
            ordered_voters: list[list[str]] = [[] for _ in range(len(POLARIZED_FILE_ORDER))]
            for i, order_element in enumerate(POLARIZED_FILE_ORDER):
                for j, (data_element, mask_element) in enumerate(zip(data_list, image_masks, strict=True)):
                    typecode = data_element.meta["TYPECODE"].value
                    obscode = data_element.meta["OBSCODE"].value
                    if typecode == order_element[:2] and obscode == order_element[2]:
                        ordered_data_list[i] = data_element
                        ordered_mask_list[i] = mask_element
                        ordered_voters[i] = voter_filenames[j]
            logger.info("Ordered files are "
                        f"{[get_base_file_name(cube) if cube is not None else None for cube in ordered_data_list]}")
            data_list = [resolve_polarization_task.submit(ordered_data_list[i:i+3])
                         for i in range(0, len(POLARIZED_FILE_ORDER), 3)]
            data_list = [entry.result() for entry in data_list]
            data_list = [j for i in data_list for j in i]
            voter_filenames = ordered_voters
            image_masks = ordered_mask_list

        default_trefoil_wcs, default_trefoil_shape = load_trefoil_wcs()
        trefoil_wcs = trefoil_wcs or default_trefoil_wcs
        trefoil_shape = trefoil_shape or default_trefoil_shape

        preprocess_trefoil_inputs(data_list, image_masks, trim_edges_px, alphas_file)

        data_list = reproject_many_flow(data_list, trefoil_wcs, trefoil_shape, rolloff_width=rolloff_width,
                                        rolloff_strength=rolloff_strength)
        data_list = [identify_bright_structures_task(cube, this_voter_filenames)
                     for cube, this_voter_filenames in zip(data_list, voter_filenames, strict=True)]
        merger = merge_many_polarized_task if polarized else merge_many_clear_task
        layers_before_merge = data_list
        output_data = merger(data_list, trefoil_wcs)

        history_src = next(d for d in data_list if d is not None)
        output_data.meta.history = history_src.meta.history
    else:
        if polarized is None:
            msg = "A polarization state must be provided"
            raise ValueError(msg)

        output_data = NDCube(
            data=np.zeros(trefoil_shape),
            uncertainty=StdDevUncertainty(np.zeros(trefoil_shape)),
            wcs=trefoil_wcs,
            meta=NormalizedMetadata.load_template("PTM" if polarized else "CTM", "2"),
        )
        output_data.meta["DATE-OBS"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        output_data.meta["DATE-BEG"] = output_data.meta["DATE-OBS"]
        output_data.meta["DATE-END"] = output_data.meta["DATE-OBS"]
        layers_before_merge = []

    centers = find_central_pixel(data_list, find_first_existing_file(data_list).wcs)
    for center, cube in zip(centers, data_list, strict=False):
        if center is None:
            continue
        cx, cy = center
        obs_no = cube.meta["OBSCODE"].value
        obs = "NFI" if obs_no == "4" else "WFI"
        output_data.meta[f"CTRX{obs}{obs_no}"] = cx
        output_data.meta[f"CTRY{obs}{obs_no}"] = cy

    finalize_output(output_data, data_list)
    output_cubes = [output_data]

    if output_filename is not None:
        output_image_task(output_data, output_filename)

    for x_cube in layers_before_merge:
        meta = NormalizedMetadata.load_template("X" + x_cube.meta["TYPECODE"].value[1] + x_cube.meta["OBSCODE"].value,
                                                level="2")
        meta.history = x_cube.meta.history
        meta.provenance = [x_cube.meta["FILENAME"].value]
        for key in ["FILEVRSN", "MOONDIST", "MOON_X", "MOON_Y"]:
            meta[key] = output_data.meta[key].value
        for key in ["OUTLIER", "BADPKTS", "DATE-OBS", "DATE-AVG", "DATE-BEG", "DATE-END", "DATE"]:
            meta[key] = x_cube.meta[key].value
        obs_no = x_cube.meta["OBSCODE"].value
        obs = "NFI" if obs_no == "4" else "WFI"
        meta[f"CTRX{obs}{obs_no}"] = output_data.meta[f"CTRX{obs}{obs_no}"].value
        meta[f"CTRY{obs}{obs_no}"] = output_data.meta[f"CTRY{obs}{obs_no}"].value
        spacecraft = SPACECRAFT_OBSCODE[obs_no]
        meta[f"HAS_{spacecraft}"] = 1
        set_spacecraft_location_to_earth(x_cube)
        output_cubes.append(NDCube(x_cube.data, meta=meta, wcs=x_cube.wcs, uncertainty=x_cube.uncertainty))

    logger.info("ending level 2 core flow")
    return output_cubes
