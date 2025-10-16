from datetime import UTC, datetime

import numpy as np
from dateutil.parser import parse as parse_datetime_str
from ndcube import NDCube
from prefect import get_run_logger

from punchbowl.data import NormalizedMetadata
from punchbowl.data.punch_io import load_many_cubes_iterable
from punchbowl.data.wcs import load_quickpunch_mosaic_wcs
from punchbowl.level3.f_corona_model import fill_nans_with_interpolation, model_fcorona_for_cube
from punchbowl.prefect import punch_flow


@punch_flow(log_prints=True)
def construct_qp_f_corona_model(filenames: list[str],
                                clip_factor: float = 3.0,
                                reference_time: str | None = None,
                                num_workers: int = 8,
                                num_loaders: int = 8,
                                fill_nans: bool = True) -> list[NDCube]:
    """Construct QuickPUNCH F corona model."""
    logger = get_run_logger()

    if reference_time is None:
        reference_time = datetime.now(UTC)
    elif isinstance(reference_time, str):
        reference_time = parse_datetime_str(reference_time)

    trefoil_wcs, trefoil_shape = load_quickpunch_mosaic_wcs()

    logger.info("construct_f_corona_background started")

    if len(filenames) == 0:
        msg = "Require at least one input file"
        raise ValueError(msg)

    filenames.sort()

    data_shape = trefoil_shape

    number_of_data_frames = len(filenames)
    data_cube = np.empty((*data_shape, number_of_data_frames), dtype=float)

    meta_list = []
    obs_times = []

    logger.info("beginning data loading")
    dates = []
    n_failed = 0
    j = 0
    for i, result in enumerate(load_many_cubes_iterable(filenames, allow_errors=True, n_workers=num_loaders)):
        if isinstance(result, str):
            logger.warning(f"Loading {filenames[i]} failed")
            logger.warning(result)
            n_failed += 1
            if n_failed > 10:
                raise RuntimeError(f"{n_failed} files failed to load, stopping")
            continue
        cube = result
        dates.append(cube.meta.datetime)
        data_cube[..., j] = np.where(np.isnan(cube.uncertainty.array), np.nan, cube.data)
        j += 1
        obs_times.append(cube.meta.datetime.timestamp())
        meta_list.append(cube.meta)
        if i % 50 == 0:
            logger.info(f"Loaded {i}/{len(filenames)} files")
    # Crop the unused end of the array if we had a few files that errored out
    data_cube = data_cube[:j+1]
    logger.info("ending data loading")
    output_datebeg = min(dates).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    output_dateend = max(dates).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    reference_xt = reference_time.timestamp()
    model_fcorona, _ = model_fcorona_for_cube(obs_times, reference_xt, data_cube,
                                              num_workers=num_workers, clip_factor=clip_factor)
    model_fcorona[model_fcorona<=0] = np.nan
    if fill_nans:
            model_fcorona = fill_nans_with_interpolation(model_fcorona)



    meta = NormalizedMetadata.load_template("CFM", "Q")

    meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    meta["DATE-AVG"] = reference_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    meta["DATE-OBS"] = reference_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    meta["DATE-BEG"] = output_datebeg
    meta["DATE-END"] = output_dateend

    output_cube = NDCube(data=model_fcorona.squeeze(),
                                meta=meta,
                                wcs=trefoil_wcs)

    return [output_cube]
