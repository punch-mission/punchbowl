import os
import pathlib
import warnings
from datetime import UTC, datetime

import numba
import numpy as np
from dateutil.parser import parse as parse_datetime
from ndcube import NDCube
from prefect import get_run_logger
from scipy.special import erfinv

from punchbowl.data import NormalizedMetadata, load_ndcube_from_fits
from punchbowl.data.punch_io import load_many_cubes_iterable
from punchbowl.exceptions import (
    CantInterpolateWarning,
    IncorrectPolarizationStateError,
    IncorrectTelescopeError,
    InvalidDataError,
)
from punchbowl.prefect import punch_flow, punch_task
from punchbowl.util import (average_datetime, bundle_matched_mzp,
                            interpolate_data, parallel_sort_first_axis,
                            masked_mean, nan_percentile)


@punch_flow
def estimate_stray_light(filepaths: list[str], # noqa: C901
                         percentile: float = 1,
                         do_uncertainty: bool = True,
                         reference_time: datetime | str | None = None,
                         exclude_percentile: float = 50,
                         erfinv_scale: float = 0.75,
                         num_workers: int | None = None,
                         num_loaders: int | None = None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Estimate the fixed stray light pattern using a percentile."""
    logger = get_run_logger()
    logger.info(f"Running with {len(filepaths)} input files")
    if isinstance(reference_time, str):
        reference_time = datetime.strptime(reference_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    data = None
    uncertainty = None
    date_obses = []
    n_failed = 0
    j = 0
    for i, result in enumerate(load_many_cubes_iterable(filepaths, n_workers=num_loaders, allow_errors=True)):
        if isinstance(result, str):
            logger.warning(f"Loading {filepaths[i]} failed")
            logger.warning(result)
            n_failed += 1
            if n_failed > 10:
                raise RuntimeError(f"{n_failed} files failed to load, stopping")
            continue
        cube = result
        date_obses.append(cube.meta.datetime)
        if data is None:
            data = np.empty((len(filepaths), *cube.data.shape))
        data[j] = cube.data
        j += 1
        if do_uncertainty:
            if uncertainty is None:
                uncertainty = np.zeros_like(cube.data)
            if cube.uncertainty is not None:
                # The final uncertainty is sqrt(sum(square(input uncertainties))), so we accumulate the squares here
                uncertainty += cube.uncertainty.array ** 2
        if (i+1) % 50 == 0:
            logger.info(f"Loaded {i+1}/{len(filepaths)} files")
    # Crop the unused end of the array if we had a few files that errored out
    data = data[:j+1]

    logger.info(f"Images loaded; they span {min(date_obses).strftime('%Y-%m-%dT%H:%M:%S')} to "
                f"{max(date_obses).strftime('%Y-%m-%dT%H:%M:%S')}")

    if num_workers:
        numba.config.NUMBA_NUM_THREADS = num_workers

    parallel_sort_first_axis(data, inplace=True)

    index_exclude = np.floor(len(filepaths) * exclude_percentile / 100).astype(int)
    index_percentile = np.floor(len(filepaths) * percentile / 100).astype(int)
    stray_light_estimate = data[index_percentile, :, :]

    stray_light_std = np.std(data[0:index_exclude, :, :], axis=0)

    sigma_offset = -1 * erfinv((-1 + percentile / 50) * erfinv_scale)

    stray_light_estimate2 = stray_light_estimate + sigma_offset * stray_light_std

    if do_uncertainty:
        uncertainty = np.sqrt(uncertainty) / len(filepaths) if do_uncertainty else None

    out_type = "S" + cube.meta.product_code[1:]
    meta = NormalizedMetadata.load_template(out_type, "1")
    meta["DATE-AVG"] = average_datetime(date_obses).strftime("%Y-%m-%dT%H:%M:%S")
    meta["DATE-OBS"] = reference_time.strftime("%Y-%m-%dT%H:%M:%S") if reference_time else meta["DATE-AVG"].value
    meta["DATE-BEG"] = min(date_obses).strftime("%Y-%m-%dT%H:%M:%S")
    meta["DATE-END"] = max(date_obses).strftime("%Y-%m-%dT%H:%M:%S")
    meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    meta.history.add_now("stray light",
                         f"Generated with {len(filepaths)} files running from "
                         f"{min(date_obses).strftime('%Y-%m-%dT%H:%M:%S')} to "
                         f"{max(date_obses).strftime('%Y-%m-%dT%H:%M:%S')}")
    meta["FILEVRSN"] = cube.meta["FILEVRSN"].value

    # Let's put in a valid, representative WCS, with the right scale and pointing, etc. But let's set the rotation to
    # zero---the rotation value is meaningless, so it should be an obvious filler value
    wcs = cube.wcs
    wcs.wcs.pc = np.eye(2)
    out_cube = NDCube(data=stray_light_estimate2, meta=meta, wcs=wcs, uncertainty=uncertainty)

    return [out_cube]

@punch_flow
def estimate_polarized_stray_light( # noqa: C901
                mfilepaths: list[str],
                zfilepaths: list[str],
                pfilepaths: list[str],
                percentile: float = 2,
                do_uncertainty: bool = True,
                reference_time: datetime | str | None = None,
                num_loaders: int | None = None,
                ) -> list[NDCube]:
    """Estimate the polarized stray light pattern using minimum indexing method."""
    logger = get_run_logger()

    if isinstance(reference_time, str):
        reference_time = datetime.strptime(reference_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)

    n_failed = 0
    mcube_list, zcube_list, pcube_list = [], [], []
    for i, result in enumerate(load_many_cubes_iterable(mfilepaths, n_workers=num_loaders, allow_errors=True,
                                                        include_provenance=False, include_uncertainty=do_uncertainty)):
        if isinstance(result, str):
            logger.warning(f"Loading {mfilepaths[i]} failed")
            logger.warning(result)
            n_failed += 1
            if n_failed > 10:
                raise RuntimeError(f"{n_failed} files failed to load, stopping")
            continue
        mcube_list.append(result)
        if (i + 1) % 50 == 0:
            logger.info(f"Loaded {i+1}/{len(mfilepaths)} M files")

    for i, result in enumerate(load_many_cubes_iterable(zfilepaths, n_workers=num_loaders, allow_errors=True,
                                                        include_provenance=False, include_uncertainty=do_uncertainty)):
        if isinstance(result, str):
            logger.warning(f"Loading {zfilepaths[i]} failed")
            logger.warning(result)
            n_failed += 1
            if n_failed > 10:
                raise RuntimeError(f"{n_failed} files failed to load, stopping")
            continue
        zcube_list.append(result)
        if (i + 1) % 50 == 0:
            logger.info(f"Loaded {i+1}/{len(zfilepaths)} Z files")

    for i, result in enumerate(load_many_cubes_iterable(pfilepaths, n_workers=num_loaders, allow_errors=True,
                                                        include_provenance=False, include_uncertainty=do_uncertainty)):
        if isinstance(result, str):
            logger.warning(f"Loading {pfilepaths[i]} failed")
            logger.warning(result)
            n_failed += 1
            if n_failed > 10:
                raise RuntimeError(f"{n_failed} files failed to load, stopping")
            continue
        pcube_list.append(result)
        if (i + 1) % 50 == 0:
            logger.info(f"Loaded {i+1}/{len(pfilepaths)} P files")

    triplets = bundle_matched_mzp(mcube_list, zcube_list, pcube_list)
    logger.info(f"Matched {len(triplets)} MZP triplets")
    # This is a RAM-intensive operation. To reduce memory usage, we'll be deleting the cubes as we iterate through
    # the list of triplets. We also need the raw cube lists gone for the cubes to be deleted
    del mcube_list, zcube_list, pcube_list

    mdata = np.empty((len(triplets), *triplets[0][0].data.shape), dtype=triplets[0][0].data.dtype)
    zdata = np.empty_like(mdata)
    pdata = np.empty_like(mdata)
    mmetas, zmetas, pmetas = [], [], []
    date_obses = []
    uncertainty = np.zeros_like(triplets[0][0].data) if do_uncertainty else None
    i = 0
    while triplets:
        # Remove the cubes from the triplets list, so they'll be deleted on the next iteration.
        mcube, zcube, pcube = triplets.pop(0)
        mdata[i] = mcube.data
        zdata[i] = zcube.data
        pdata[i] = pcube.data
        mmetas.append(mcube.meta)
        zmetas.append(zcube.meta)
        pmetas.append(pcube.meta)
        mwcs, zwcs, pwcs = mcube.wcs, zcube.wcs, pcube.wcs

        date_obses.extend([cube.meta.datetime for cube in (mcube, zcube, pcube)])

        if do_uncertainty:
            for cube in [mcube, zcube, pcube]:
                if cube.uncertainty is not None:
                    uncertainty += cube.uncertainty.array ** 2

        i += 1
    # triplets is now empty and useless. To be clear about that, let's delete it
    del triplets

    logger.info(f"Images loaded; they span {min(date_obses).strftime('%Y-%m-%dT%H:%M:%S')} to "
                f"{max(date_obses).strftime('%Y-%m-%dT%H:%M:%S')}")

    # Estimate total brightness
    tbcube = 2 / 3 * (mdata + zdata + pdata)

    # Per-pixel percentile threshold of tbcube over time (T axis)
    tb_thresh = nan_percentile(tbcube, percentile, axis=0, keepdims=True)
    mask = (tbcube <= tb_thresh)  # shape: (T, H, W)

    # Estimate MZP background based on index
    m_background = masked_mean(mdata, mask)
    z_background = masked_mean(zdata, mask)
    p_background = masked_mean(pdata, mask)

    if do_uncertainty:
        uncertainty = np.sqrt(uncertainty) / len(mfilepaths)

    output_cubes = []
    for label, background, metas, sample_wcs in zip(
        ["M", "Z", "P"], [m_background, z_background, p_background],
            (mmetas, zmetas, pmetas), (mwcs, zwcs, pwcs), strict=True):
        out_type = "S" + label
        meta = NormalizedMetadata.load_template(out_type + metas[0]["OBSCODE"].value, "1")

        meta.provenance = [meta["FILENAME"].value for meta in metas]
        meta["DATE-AVG"] = average_datetime(date_obses).strftime("%Y-%m-%dT%H:%M:%S")
        meta["DATE-OBS"] = reference_time.strftime("%Y-%m-%dT%H:%M:%S") \
            if reference_time else meta["DATE-AVG"].value
        meta["DATE-BEG"] = min(date_obses).strftime("%Y-%m-%dT%H:%M:%S")
        meta["DATE-END"] = max(date_obses).strftime("%Y-%m-%dT%H:%M:%S")
        meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")

        meta.history.add_now(
            "polarized stray light",
            f"Generated with {len(metas)} files running from "
            f"{min(date_obses).strftime('%Y-%m-%dT%H:%M:%S')} to "
            f"{max(date_obses).strftime('%Y-%m-%dT%H:%M:%S')}")
        meta["FILEVRSN"] = metas[0]["FILEVRSN"].value

        wcs = sample_wcs.deepcopy()
        wcs.wcs.pc = np.eye(2)

        output_cubes.append(NDCube(data=background, meta=meta, wcs=wcs, uncertainty=uncertainty))

    return output_cubes

@punch_task
def remove_stray_light_task(data_object: NDCube, #noqa: C901
                            stray_light_before_path: pathlib.Path | str | NDCube,
                            stray_light_after_path: pathlib.Path | str | NDCube) -> NDCube:
    """
    Prefect task to remove stray light from an image.

    Stray light is light in an optical system which was not intended in the
    design.

    The PUNCH instrument stray light will be mapped periodically as part of the
    ongoing in-flight calibration effort. The stray light maps will be
    generated directly from the L0 and L1 science data. Separating instrumental
    stray light from the F-corona. This has been demonstrated with SOHO/LASCO
    and with STEREO/COR2 observations. It requires an instrumental roll to hold
    the stray light pattern fixed while the F-corona rotates in the field of
    view. PUNCH orbital rolls will be used to create similar effects.

    Uncertainty across the image plane is calculated using a known stray light
    model and the difference between the calculated stray light and the ground
    truth. The uncertainty is convolved with the input uncertainty layer to
    produce the output uncertainty layer.


    Parameters
    ----------
    data_object : NDCube
        data to operate on

    stray_light_before_path: pathlib
        path to stray light model before observation to apply to data

    stray_light_after_path: pathlib
        path to stray light model after observation to apply to data

    Returns
    -------
    NDCube
        modified version of the input with the stray light removed

    """
    if stray_light_before_path is None or stray_light_after_path is None:
        data_object.meta.history.add_now("LEVEL1-remove_stray_light", "Stray light correction skipped")
        return data_object

    if isinstance(stray_light_before_path, NDCube):
        stray_light_before_model = stray_light_before_path
        stray_light_before_path = stray_light_before_model.meta["FILENAME"].value
    else:
        stray_light_before_path = pathlib.Path(stray_light_before_path)
        if not stray_light_before_path.exists():
            msg = f"File {stray_light_before_path} does not exist."
            raise InvalidDataError(msg)
        stray_light_before_model = load_ndcube_from_fits(stray_light_before_path)
        stray_light_before_path = stray_light_before_model.meta["FILENAME"].value

    if isinstance(stray_light_after_path, NDCube):
        stray_light_after_model = stray_light_after_path
        stray_light_after_path = stray_light_after_model.meta["FILENAME"].value
    else:
        stray_light_after_path = pathlib.Path(stray_light_after_path)
        if not stray_light_after_path.exists():
            msg = f"File {stray_light_after_path} does not exist."
            raise InvalidDataError(msg)
        stray_light_after_model = load_ndcube_from_fits(stray_light_after_path)

    for model in stray_light_before_model, stray_light_after_model:
        if model.meta["TELESCOP"].value != data_object.meta["TELESCOP"].value:
            msg=f"Incorrect TELESCOP value within {model['FILENAME'].value}"
            raise IncorrectTelescopeError(msg)
        if model.meta["OBSLAYR1"].value != data_object.meta["OBSLAYR1"].value:
            msg=f"Incorrect polarization state within {model['FILENAME'].value}"
            raise IncorrectPolarizationStateError(msg)
        if model.data.shape != data_object.data.shape:
            msg = f"Incorrect stray light function shape within {model['FILENAME'].value}"
            raise InvalidDataError(msg)

    # For the quickpunch case, our stray light models run right up to the current time, with their DATE-OBS likely days
    # in the past. It feels reckless to interpolate the six-hour variation in the model over several days, so let's
    # instead interpolate using the nearst of DATE-BEG, DATE-AVG, or DATE-END. (DATE-BEG will be the best choice when
    # reprocessing.)
    delta_dateavg = abs(parse_datetime(stray_light_before_model.meta["DATE-AVG"].value + " UTC")
                        - data_object.meta.datetime)
    delta_datebeg = abs(parse_datetime(stray_light_before_model.meta["DATE-BEG"].value + " UTC")
                        - data_object.meta.datetime)
    delta_dateend = abs(parse_datetime(stray_light_before_model.meta["DATE-END"].value + " UTC")
                        - data_object.meta.datetime)

    closest = min(delta_datebeg, delta_dateavg, delta_dateend)
    if closest is delta_datebeg:
        time_key = "DATE-BEG"
    elif closest is delta_dateavg:
        time_key = "DATE-AVG"
    else:
        time_key = "DATE-END"

    if stray_light_before_model.meta[time_key].value == stray_light_after_model.meta[time_key].value:
        warnings.warn(
            "Timestamps are identical for the stray light models; can't inter/extrapolate", CantInterpolateWarning)
        stray_light_model = stray_light_before_model.data
    else:
        stray_light_model = interpolate_data(stray_light_before_model,
                                             stray_light_after_model,
                                             data_object.meta.datetime,
                                             time_key=time_key,
                                             allow_extrapolation=True)
    data_object.data[:, :] -= stray_light_model
    uncertainty = 0
    # TODO: when we have real uncertainties, use them
    # uncertainty = stray_light_model.uncertainty.array # noqa: ERA001
    data_object.uncertainty.array[...] = np.sqrt(data_object.uncertainty.array**2 + uncertainty**2)
    data_object.meta.history.add_now("LEVEL1-remove_stray_light",
                                     f"stray light removed with {os.path.basename(str(stray_light_before_path))} "
                                     f"and {os.path.basename(str(stray_light_after_path))}")
    return data_object
