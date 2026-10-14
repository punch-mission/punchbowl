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
    """
    Level 3 CTM/PTM flow.

    Parameters
    ----------
    data_list : list[str | PUNCHCube]
        The input data files---*IMs to process into *TMs
    nfi_list : list[str | PUNCHCube | None]
        NFI files, if any, to insert in the mosaics.
    before_starfield_path : str | None
        The path to a starfield model before the observations
    after_starfield_path : str | None
        The path to a starfield model after the observations
    nfi_wfi_divide_radius : float
        A radius in pixels that divides the NFI region of the final mosiac from the WFI region. If not provided, there
        will be no NFI region.
    nfi_scale_factor : float
        An ad-hoc scale factor to apply to the NFI pixel values.
    output_filename : str | None
        A location at which to save the output file

    Returns
    -------
    out_data_list : list[PUNCHCube]
        The *TM cubes produced by this flow

    """
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

    nfi_wfi_mask = (make_circular_mask(data_list[0].shape, nfi_wfi_divide_radius)
                    if nfi_wfi_divide_radius is not None else None)

    out_data_list = []
    for wfi_cube, nfi_cube in zip(data_list, nfi_list, strict=True):
        out_meta: NormalizedMetadata = NormalizedMetadata.load_template("PTM" if is_polarized else "CTM", "3")

        if nfi_cube is not None and nfi_wfi_mask is not None:
            # Paste NFI into the defined region
            wfi_cube.data[:] = np.where(nfi_wfi_mask, nfi_scale_factor * nfi_cube.data, wfi_cube.data)
            wfi_cube.uncertainty.array[:] = np.where(
                nfi_wfi_mask, nfi_cube.uncertainty.array, wfi_cube.uncertainty.array)
            wfi_cube.mask[:] = np.isinf(wfi_cube.uncertainty.array)
            out_meta["OUTLIER"] = wfi_cube.meta["OUTLIER"].value | encode_outliers([nfi_cube])
            out_meta["HAS_NFI4"] = 1
            out_meta["CTRXNFI4"] = 2047.5
            out_meta["CTRYNFI4"] = 2047.5
            out_meta["ALL_INPT"] = wfi_cube.meta["ALL_INPT"].value
        else:
            if nfi_wfi_mask is not None:
                # We've been given a radius for a NFI-only region, but we don't have NFI data. Make sure that region
                # gets cleared and marked properly.
                wfi_cube.data[:] = np.where(nfi_wfi_mask, 0, wfi_cube.data)
                wfi_cube.uncertainty.array[:] = np.where(
                    nfi_wfi_mask, np.inf, wfi_cube.uncertainty.array)
            out_meta["ALL_INPT"] = 0
            out_meta["HAS_NFI4"] = 0

        out_meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        out_meta.provenance = [wfi_cube.meta["FILENAME"].value]
        if nfi_cube is not None and nfi_wfi_mask is not None:
            out_meta.provenance += [nfi_cube.meta["FILENAME"].value]
        out_meta.history = wfi_cube.meta.history
        out_meta["CALSTAR1"] = before_starfield_path
        out_meta["CALSTAR2"] = after_starfield_path
        for key in ["FILEVRSN", "HAS_WFI1", "HAS_WFI2", "HAS_WFI3", "DATE-AVG", "DATE-OBS",
                    "DATE-BEG", "DATE-END", "CTRXWFI1", "CTRYWFI1", "CTRXWFI2", "CTRYWFI2", "CTRXWFI3", "CTRYWFI3",
                    "CTRXNFI4", "CTRYNFI4"]:
            out_meta[key] = wfi_cube.meta[key].value
        output_data = wfi_cube.replace(meta=out_meta)
        output_data = set_spacecraft_location_to_earth(output_data)
        out_data_list.append(output_data)

    if output_filename is not None:
        output_image_task(out_data_list[0], output_filename)

    logger.info("ending level 3 core flow")

    return out_data_list


@punch_flow
def generate_level3_low_noise_flow(data_list: list[str | PUNCHCube],
                                   nfi_list: list[str | PUNCHCube] | None,
                                   output_filename: str | None = None,
                                   nfi_wfi_divide_radius: float | None = None,
                                   nfi_scale_factor: float = 1,
                                   reference_time: str | datetime | None = None) -> list[PUNCHCube]:
    """Generate low noise products."""
    logger = get_logger()

    logger.info("Generating low noise products")
    data_list = [load_image_task(d) if isinstance(d, str) else d for d in data_list]
    if nfi_list and nfi_wfi_divide_radius:
        nfi_list = [load_image_task(d) if isinstance(d, str) else d for d in nfi_list]
    low_noise_image = create_low_noise_task(data_list, reference_time=reference_time, nfi_cubes=nfi_list,
                                            nfi_wfi_divide_radius=nfi_wfi_divide_radius,
                                            nfi_scale_factor=nfi_scale_factor)

    if output_filename is not None:
        output_image_task(low_noise_image, output_filename)

    return [low_noise_image]


@punch_flow
def generate_level3_velocity_flow(files: list[str],
                                  product_code: str = "PTM",
                                  frames_per_window: int | None = None,
                                  delta_t: int | None = None,
                                  sparsity: int | None = None,
                                  n_ofs: int | None = None,
                                  delta_px: int | None = None,
                                  expected_wind_kps: int | None = None,
                                  offset_speed_kps: int | None = None,
                                  annuli_centers_rs: list[float] | np.ndarray | None = None,
                                  annuli_width_rs: float | None = None,
                                  azimuth_bins_remap: int | None = None,
                                  az_bin: int | None = None,
                                  vel_bin_width: int | None = None,
                                  speed_max: int | None = None,
                                  buffer_target_hours: int | None = None,
                                  use_median: bool | None = None,
                                  rotate90: bool | None = None,
                                  remove_temporal_median: bool | None = None,
                                  deflicker: bool | None = None,
                                  hdu: int | None = None,
                                  output_filename: str | None = None) -> list[PUNCHCube]:
    """
    Generate level 3 flow tracking velocity product.

    All optional flow-tracking parameters follow the signature of
    :func:`punchbowl.level3.velocity.track_velocity`. Any parameter left as
    ``None`` here is not forwarded, so the authoritative defaults defined by
    ``track_velocity`` itself are used.

    Parameters
    ----------
    files : list[str]
        Input files used for velocity tracking
    product_code : str, optional
        PUNCH data product code of the input files ("CAM", "PAM", "CTM" or "PTM"),
        by default "PTM" (VAMs are generated from PTM time series)
    frames_per_window : int, optional
        Number of frames in the tracking window
    delta_t : int, optional
        Time offset in frames between images
    sparsity : int, optional
        Frame skip interval for averaging
    n_ofs : int, optional
        Number of spatial offsets for cross-correlation
    delta_px : int, optional
        Pixel offset increment per sample
    expected_wind_kps : int, optional
        Expected wind speed in km/s
    offset_speed_kps : int, optional
        Offset speed before which speeds are ignored when searching for the peak speed
    annuli_centers_rs : list[float], optional
        Centers of the annuli, in solar radii
    annuli_width_rs : float, optional
        Width of the annuli in solar radii
    azimuth_bins_remap : int, optional
        Number of azimuthal bins in the polar-remapped images before binning
    az_bin : int, optional
        Binning factor over azimuth of the polar-remapped images
    vel_bin_width : int, optional
        Bin size over azimuth of the output flow maps
    speed_max : int, optional
        Maximum velocity to consider in the moments and peak calculation
    buffer_target_hours : int, optional
        Minimum duration for the buffer used to calculate temporal averages
    use_median : bool, optional
        Whether to use the median instead of the mean for the temporal averages
    rotate90 : bool, optional
        Whether to rotate the output flow map by 90 degrees to get solar north
        as origin of position angle
    remove_temporal_median : bool, optional
        Whether to remove the temporal median from the images
    deflicker : bool, optional
        Whether to apply an azimuthal filter to attenuate flickering
    hdu : int, optional
        Position of the header data unit in the FITS files holding the data
    output_filename : str, optional
        Output file name, by default None

    Returns
    -------
    list[PUNCHCube]
        List of generated velocity maps

    """
    logger = get_logger()

    logger.info("Generating velocity data product")
    velocity_options = {
        "frames_per_window": frames_per_window,
        "delta_t": delta_t,
        "sparsity": sparsity,
        "n_ofs": n_ofs,
        "delta_px": delta_px,
        "expected_wind_kps": expected_wind_kps,
        "offset_speed_kps": offset_speed_kps,
        "annuli_centers_rs": annuli_centers_rs,
        "annuli_width_rs": annuli_width_rs,
        "azimuth_bins_remap": azimuth_bins_remap,
        "az_bin": az_bin,
        "vel_bin_width": vel_bin_width,
        "speed_max": speed_max,
        "buffer_target_hours": buffer_target_hours,
        "use_median": use_median,
        "rotate90": rotate90,
        "remove_temporal_median": remove_temporal_median,
        "deflicker": deflicker,
        "hdu": hdu,
    }
    velocity_kwargs = {key: value for key, value in velocity_options.items() if value is not None}
    velocity_data = track_velocity(files=files, product_code=product_code, **velocity_kwargs)

    if output_filename is not None:
        # The VAM carries a custom azimuth/radius WCS that cannot be converted to
        # a celestial WCS, so skip the conversion when writing it out.
        output_image_task(velocity_data, output_filename, skip_wcs_conversion=True)
        plot_filename = f"{os.path.splitext(output_filename)[0]}.png"
        plot_rebin = 10 if velocity_data.data.shape[1] % 10 == 0 else 1
        plot_flow_map(velocity_data, rebin=plot_rebin, filename=plot_filename)

    return [velocity_data]
