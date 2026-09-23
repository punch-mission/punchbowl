import os
from copy import deepcopy
from datetime import UTC, datetime

import numpy as np
from astropy.nddata import StdDevUncertainty

from punchbowl.auto.control.cache_layer.loader_base_class import DataLoader
from punchbowl.data import load_ndcube_from_fits, load_trefoil_wcs
from punchbowl.data.meta import MetaField, NormalizedMetadata, check_moon_in_fov, set_spacecraft_location_to_earth
from punchbowl.data.punch_io import encode_outliers
from punchbowl.data.punchcube import PUNCHCube
from punchbowl.level2.finalize import finalize_output
from punchbowl.level2.merge import merge_many_clear_task, merge_many_polarized_task
from punchbowl.level2.resample import reproject_cube
from punchbowl.level3.f_corona_model import subtract_f_corona_background_task
from punchbowl.level3.low_noise import create_low_noise_task
from punchbowl.level3.polarization import convert_polarization
from punchbowl.level3.stellar import subtract_starfield_background_task
from punchbowl.level3.velocity import plot_flow_map, track_velocity
from punchbowl.prefect import get_logger, punch_flow
from punchbowl.util import load_image_task, make_circular_mask, output_image_task


@punch_flow
def level3_NFI_flow(data_list: list[str | PUNCHCube],  # noqa: N802
                    before_f_corona_model_path: str | DataLoader,
                    after_f_corona_model_path: str | DataLoader,
                    inner_mask_radius: float,
                    outer_mask_radius: float) -> list[PUNCHCube]:
    """
    Run Level 3 NFI F-corona subtraction and reprojection flow.

    L2 NFI images have an F-corona model subtracted and are reprojected to the full-mosaic frame. Returns both the
    full-res image (as an L3 CNN) and the mosaic-frame (as an XR4).

    Parameters
    ----------
    data_list : list[str | PUNCHCube]
        The images to process
    before_f_corona_model_path : str | DataLoader
        The first F corona model
    after_f_corona_model_path : str | DataLoader
        The second F corona model
    inner_mask_radius : float
        The mosaic-frame image will be cropped at this inner radius
    outer_mask_radius : float
        The mosaic-frame image will be cropped at this outer radius

    Returns
    -------
    list[PUNCHCube]
        The output data cubes

    """
    logger = get_logger()

    logger.info("beginning level 3 NFI flow")
    data_list = [load_image_task(d) if isinstance(d, str) else d for d in data_list]

    if isinstance(before_f_corona_model_path, str):
        before_f_corona_model = load_ndcube_from_fits(before_f_corona_model_path)
    else:
        before_f_corona_model = before_f_corona_model_path.load()
        before_f_corona_model_path = before_f_corona_model_path.src_repr()

    if isinstance(after_f_corona_model_path, str):
        after_f_corona_model = load_ndcube_from_fits(after_f_corona_model_path)
    else:
        after_f_corona_model = after_f_corona_model_path.load()
        after_f_corona_model_path = after_f_corona_model_path.src_repr()


    mosaic_wcs, mosaic_shape = load_trefoil_wcs()
    inner_mask = ~make_circular_mask(mosaic_shape, inner_mask_radius)
    outer_mask = make_circular_mask(mosaic_shape, outer_mask_radius)
    mask = inner_mask * outer_mask

    output_cubes = []
    for cube in data_list:
        cube = subtract_f_corona_background_task(cube, [before_f_corona_model], [after_f_corona_model]) # noqa: PLW2901
        mosaic_data, mosaic_uncert = reproject_cube(cube, mosaic_wcs, mosaic_shape, rolloff_strength=0, rolloff_width=0)
        np.nan_to_num(mosaic_data, copy=False)
        np.nan_to_num(mosaic_uncert, copy=False, nan=np.inf)
        mosaic_cube = PUNCHCube(data=mosaic_data, uncertainty=StdDevUncertainty(mosaic_uncert), wcs=mosaic_wcs,
                                meta=cube.meta)
        mosaic_cube.data *= mask
        mosaic_cube.uncertainty.array[~mask] = np.inf

        new_meta = NormalizedMetadata.load_template("CNN", "3")
        new_meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        for key in cube.meta:
            if ((key in ["DATE-OBS", "DATE-BEG", "DATE-AVG", "DATE-END", "FILEVRSN", "OUTLIER", "BADPKTS", "OUTLIER",
                         "XACTTIME", "GEOD_LON", "GEOD_LAT", "GEOD_ALT", "LOS_ALT"]
                    or key[-4:] in ["_OBS", "_VOB"]
                    or key[:3] in ("PCA", "CAL"))
                    and key in new_meta):
                new_meta[key] = cube.meta[key].value
        _, _, _, _, moondist, xpix, ypix = check_moon_in_fov(
            cube.meta["DATE-OBS"].value, wcs=cube.wcs, image_shape=cube.data.shape)
        new_meta["MOONDIST"] = moondist[0]
        new_meta["MOON_X"] = xpix[0]
        new_meta["MOON_Y"] = ypix[0]
        new_meta["CALFCOR1"] = os.path.basename(before_f_corona_model_path)
        new_meta["CALFCOR2"] = os.path.basename(after_f_corona_model_path)
        new_meta["CTRXNFI4"] = cube.wcs.wcs.crpix[1] - 1
        new_meta["CTRYNFI4"] = cube.wcs.wcs.crpix[0] - 1

        new_meta.provenance = [cube.meta["FILENAME"].value]

        cube = cube.replace(meta=new_meta) # noqa: PLW2901
        output_cubes.append(cube)

        new_meta = NormalizedMetadata.load_template("XR4", "3")
        new_meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        for key in mosaic_cube.meta:
            if ((key in ["DATE-OBS", "DATE-BEG", "DATE-AVG", "DATE-END", "FILEVRSN", "OUTLIER", "BADPKTS", "OUTLIER",
                         "XACTTIME", "GEOD_LON", "GEOD_LAT", "GEOD_ALT", "LOS_ALT"]
                    or key[-4:] in ["_OBS", "_VOB"]
                    or key[:3] in ("PCA", "CAL"))
                    and key in new_meta):
                new_meta[key] = mosaic_cube.meta[key].value
        new_meta["CALFCOR1"] = os.path.basename(before_f_corona_model_path)
        new_meta["CALFCOR2"] = os.path.basename(after_f_corona_model_path)
        new_meta["CTRXNFI4"] = mosaic_cube.wcs.wcs.crpix[1] - 1
        new_meta["CTRYNFI4"] = mosaic_cube.wcs.wcs.crpix[0] - 1

        new_meta.provenance = [mosaic_cube.meta["FILENAME"].value]

        mosaic_cube = mosaic_cube.replace(meta=new_meta)
        output_cubes.append(mosaic_cube)

    logger.info("ending level 3 NFI flow")

    return output_cubes


@punch_flow
def level3_PIM_CIM_flow(data_list: list[str] | list[PUNCHCube],  # noqa: N802
                        before_f_corona_model_paths: list[str | DataLoader],
                        after_f_corona_model_paths: list[str | DataLoader],
                        output_filename: str | None = None) -> list[PUNCHCube]:
    """Level 3 PIM/CIM flow."""
    logger = get_logger()

    logger.info("beginning level 3 PIM/CIM flow")
    data_list = [load_image_task(d) if isinstance(d, str) else d for d in data_list]
    for i, cube in enumerate(data_list):
        if len(cube.shape) == 3:
            data = np.full((cube.shape[0], cube.meta["FULYSIZE"].value, cube.meta["FULXSIZE"].value), np.nan)
        else:
            data = np.full((cube.meta["FULYSIZE"].value, cube.meta["FULXSIZE"].value), np.nan)
        cropx = cube.meta["CROPX1"].value, cube.meta["CROPX2"].value
        cropy = cube.meta["CROPY1"].value, cube.meta["CROPY2"].value
        data[..., cropy[0]:cropy[1], cropx[0]:cropx[1]] = cube.data
        uncertainty = np.full(data.shape, np.inf)
        uncertainty[..., cropy[0]:cropy[1], cropx[0]:cropx[1]] = cube.uncertainty.array
        new_cube = cube.replace(data=data, uncertainty=uncertainty)
        data_list[i] = new_cube
    polarized = data_list[0].meta["TYPECODE"].value[1] != "R"
    new_type = "PIM" if polarized else "CIM"
    trefoil_wcs = data_list[0].wcs.celestial

    before_f_corona_models = [load_ndcube_from_fits(path) if isinstance(path, str)
                              else path.load() for path in before_f_corona_model_paths]
    after_f_corona_models = [load_ndcube_from_fits(path) if isinstance(path, str)
                              else path.load()  for path in after_f_corona_model_paths]

    data_list = [subtract_f_corona_background_task(d,
                                                   before_f_corona_models,
                                                   after_f_corona_models) for d in data_list]

    if polarized:
        merge_layers = []
        # The merging code wants our layers separated out as individual cubes
        for d in data_list:
            if d is None:
                continue
            for i, angle in enumerate([-60, 0, 60]):
                # The input cubes need to have "POLAR" set so it knows which layer is which
                m = deepcopy(d.meta)
                # The existing meta doesn't have a POLAR key. Hack: just grab a section and cram in the new value.
                section = next(iter(m._contents.values())) # noqa: SLF001
                section["POLAR"] = MetaField("POLAR", "", angle, int, True, True, 0)
                merge_layers.append(PUNCHCube(
                    d.data[i],
                    meta=m,
                    wcs=d.wcs,
                    uncertainty=d.uncertainty[i],
                ))
    else:
        merge_layers = data_list
    merger = merge_many_polarized_task if polarized else merge_many_clear_task
    output_data = merger(merge_layers, trefoil_wcs, level="3", product_code=new_type)
    fcor_files = [c.meta["FILENAME"].value.replace(".fits", "") for c in before_f_corona_models + after_f_corona_models]
    output_data.meta.history.add_now("LEVEL3-subtract_f_corona_background",
                                     f"subtracted f corona background using {', '.join(fcor_files)}")

    finalize_output(output_data, data_list)

    for cube in data_list:
        obs_no = cube.meta["OBSCODE"].value
        obs = "NFI" if obs_no == "4" else "WFI"
        if cube.meta[f"CTRX{obs}{obs_no}"].value > 0:
            output_data[0].meta[f"CTRX{obs}{obs_no}"] = cube.meta[f"CTRX{obs}{obs_no}"].value
            output_data[0].meta[f"CTRY{obs}{obs_no}"] = cube.meta[f"CTRY{obs}{obs_no}"].value

    logger.info("ending level 3 PIM/CIM flow")

    if output_filename is not None:
        output_image_task(output_data, output_filename)

    return [output_data]


@punch_flow
def level3_core_flow(data_list: list[str | PUNCHCube],
                     nfi_list: list[str | PUNCHCube | None],
                     before_starfield_path: str | None,
                     after_starfield_path: str | None,
                     nfi_wfi_divide_radius: float | None,
                     nfi_scale_factor: float = 1,
                     output_filename: str | None = None) -> list[PUNCHCube]:
    """Level 3 CTM flow."""
    logger = get_logger()

    logger.info("beginning level 3 flow")
    data_list = [load_image_task(d) if isinstance(d, str) else d for d in data_list]
    nfi_list = [load_image_task(d) if isinstance(d, str) else d for d in nfi_list]
    is_polarized = data_list[0].meta["TYPECODE"].value == "PI"
    data_list = [subtract_starfield_background_task(d,
                                                    before_starfield_path,
                                                    after_starfield_path,
                                                    is_polarized=is_polarized) for d in data_list]
    if is_polarized:
        data_list = [convert_polarization(d) for d in data_list]

    mask = make_circular_mask(data_list[0].shape, nfi_wfi_divide_radius) if nfi_wfi_divide_radius is not None else None

    out_data_list = []
    for wfi_cube, nfi_cube in zip(data_list, nfi_list, strict=True):
        out_meta: NormalizedMetadata = NormalizedMetadata.load_template("PTM" if is_polarized else "CTM", "3")

        if nfi_cube is not None and mask is not None:
            wfi_cube.data[:] = np.where(mask, nfi_scale_factor * nfi_cube.data, wfi_cube.data)
            wfi_cube.uncertainty.array[:] = np.where(mask, nfi_cube.uncertainty.array, wfi_cube.uncertainty.array)
            out_meta["OUTLIER"] = wfi_cube.meta["OUTLIER"].value | encode_outliers([nfi_cube])

        out_meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        out_meta.provenance = [wfi_cube.meta["FILENAME"].value]
        if nfi_cube is not None and mask is not None:
            out_meta.provenance += [nfi_cube.meta["FILENAME"].value]
        out_meta.history = wfi_cube.meta.history
        out_meta["CALSTAR1"] = before_starfield_path
        out_meta["CALSTAR2"] = after_starfield_path
        for key in ["FILEVRSN", "ALL_INPT", "HAS_WFI1", "HAS_WFI2", "HAS_WFI3", "HAS_NFI4", "DATE-AVG", "DATE-OBS",
                    "DATE-BEG", "DATE-END", "CTRXWFI1", "CTRYWFI1", "CTRXWFI2", "CTRYWFI2", "CTRXWFI3", "CTRYWFI3",
                    "CTRXNFI4", "CTRYNFI4"]:
            out_meta[key] = wfi_cube.meta[key].value
        if nfi_cube:
            out_meta["HAS_NFI4"] = True
            out_meta["CTRXNFI4"] = 2047.5
            out_meta["CTRYNFI4"] = 2047.5
        else:
            out_meta["ALL_INPT"] = False
        output_data = wfi_cube.replace(meta=out_meta)
        output_data = set_spacecraft_location_to_earth(output_data)
        out_data_list.append(output_data)

    if output_filename is not None:
        output_image_task(out_data_list[0], output_filename)

    logger.info("ending level 3 core flow")

    return out_data_list


@punch_flow
def generate_level3_low_noise_flow(data_list: list[str] | list[PUNCHCube],
                                   output_filename: str | None = None,
                                   reference_time: str | datetime | None = None) -> list[PUNCHCube]:
    """Generate low noise products."""
    logger = get_logger()

    logger.info("Generating low noise products")
    data_list = [load_image_task(d) if isinstance(d, str) else d for d in data_list]
    low_noise_image = create_low_noise_task(data_list, reference_time=reference_time)

    if output_filename is not None:
        output_image_task(low_noise_image, output_filename)

    return [low_noise_image]


@punch_flow
def generate_level3_velocity_flow(files: list[str],
                                  delta_t: int = 12,
                                  sparsity: int = 2,
                                  n_ofs: int = 151,
                                  ycens: np.ndarray | None = None,
                                  rbands: list[int] | None = None,
                                  output_filename: str | None = None) -> list[PUNCHCube]:
    """
    Generate level 3 flow tracking velocity product.

    Parameters
    ----------
    files : list[str]
        Input files used for velocity tracking
    delta_t : int, optional
        Time offset in frames between images, by default 12
    sparsity : int, optional
        Frame skip interval for averaging, by default 2
    n_ofs : int, optional
        Number of spatial offsets for cross-correlation, by default 151
    ycens : np.ndarray | None, optional
        Radial band centers in solar radii, by default None
    rbands : list[int] | None, optional
        Indices of radial bands to visualize, by default None
    output_filename : str | None, optional
        Output file name, by default None

    Returns
    -------
    list[PUNCHCube]
        List of generated velocity maps

    """
    logger = get_logger()

    logger.info("Generating velocity data product")
    velocity_data = track_velocity(files=files,
                                   delta_t=delta_t,
                                   sparsity=sparsity,
                                   n_ofs=n_ofs,
                                   ycens=ycens,
                                   rbands=rbands)

    if output_filename is not None:
        output_image_task(velocity_data, output_filename)
        plot_filename = f"{os.path.splitext(output_filename)[0]}.png"
        plot_flow_map(plot_filename, velocity_data)

    return [velocity_data]
