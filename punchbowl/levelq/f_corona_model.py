from datetime import UTC, datetime

import numpy as np
from dateutil.parser import parse as parse_datetime_str
from ndcube import NDCube
from prefect import get_run_logger

from punchbowl.data import NormalizedMetadata, load_ndcube_from_fits
from punchbowl.data.wcs import load_quickpunch_mosaic_wcs
from punchbowl.level3.f_corona_model import fill_nans_with_interpolation, model_fcorona_for_cube
from punchbowl.prefect import punch_flow


@punch_flow(log_prints=True)
def construct_qp_f_corona_model(filenames: list[str], smooth_level: float = 3.0,
                                       reference_time: str | None = None) -> list[NDCube]:
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
    data_cube = np.empty((number_of_data_frames, *data_shape), dtype=float)
    uncertainty_cube = np.empty((number_of_data_frames, *data_shape), dtype=float)

    meta_list = []
    obs_times = []

    logger.info("beginning data loading")
    for i, address_out in enumerate(filenames):
        data_object = load_ndcube_from_fits(address_out)
        data_cube[i, ...] = data_object.data
        uncertainty_cube[i, ...] = data_object.uncertainty.array
        obs_times.append(data_object.meta.datetime.timestamp())
        meta_list.append(data_object.meta)
    logger.info("ending data loading")

    reference_xt = reference_time.timestamp()
    model_fcorona, _ = model_fcorona_for_cube(obs_times, reference_xt, data_cube, clip_factor=smooth_level)
    model_fcorona[model_fcorona<=0] = np.nan
    model_fcorona = fill_nans_with_interpolation(model_fcorona)

    meta = NormalizedMetadata.load_template("CFM", "Q")
    meta["DATE-OBS"] = str(reference_time)
    output_cube = NDCube(data=model_fcorona.squeeze(),
                                meta=meta,
                                wcs=trefoil_wcs)

    return [output_cube]
