from datetime import UTC, datetime

import numpy as np
from astropy.nddata import StdDevUncertainty
from dateutil.parser import ParserError
from dateutil.parser import parse as parse_datetime
from dateutil.parser import parse as parse_datetime_str

from punchbowl.data.meta import NormalizedMetadata
from punchbowl.data.punch_io import check_outlier
from punchbowl.data.punchcube import PUNCHCube
from punchbowl.level2.merge import _merge_ndcubes
from punchbowl.prefect import punch_task
from punchbowl.util import average_datetime, make_circular_mask, nan_percentile

KEYWORD_OMIT = ("COMMENT", "HISTORY", "", "NAXIS3", "OBSTYPE", "OBS-MODE", "OBSLAYR1", "OBSLAYR2", "OBSLAYR3")

def merge_polarized_low_noise(cubes: list[PUNCHCube], reference_cube_index: int) -> PUNCHCube:
    """
    Merge Level 3 PTM inputs for PAM with independent propagated errors.

    With w = 1/sigma**2, the mean is sum(w*I)/sum(w) and its standard
    deviation is sqrt(1/sum(w)). Shared calibration/background covariance
    is not included.
    """
    if any(cube.meta.product_code != "PTM" or cube.meta["LEVEL"].value != "3" for cube in cubes):
        raise ValueError("The polarized low-noise merger requires Level 3 PTM inputs")
    if any(cube.uncertainty is None for cube in cubes):
        raise ValueError("PAM generation requires propagated polarization uncertainties")
    data_stack = np.stack([cube.data for cube in cubes], axis=0)
    uncertainty_stack = np.array([cube.uncertainty.array for cube in cubes], dtype=float)

    # Keep valid zero signals; exclude invalid or masked samples from the weights.
    uncertainty_stack_was_bad = (uncertainty_stack <= 0) | ~np.isfinite(uncertainty_stack) | ~np.isfinite(data_stack) | (uncertainty_stack == 0)
    for i, cube in enumerate(cubes):
        if cube.mask is not None:
            uncertainty_stack_was_bad[i] |= np.broadcast_to(cube.mask, cube.shape)
    uncertainty_stack[uncertainty_stack_was_bad] = np.inf

    weight_stack = 1/np.square(uncertainty_stack)
    total_weight = np.nansum(weight_stack, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        new_data = np.nansum(data_stack * weight_stack, axis=0) / total_weight
        final_uncertainty = np.sqrt(1 / total_weight)

    # Restores uncertainty of zero or nan to infinite
    final_uncertainty[final_uncertainty == 0] = np.inf
    final_uncertainty[~np.isfinite(final_uncertainty)] = np.inf
    new_data[np.isnan(new_data)] = 0
    return PUNCHCube(data=new_data, uncertainty=StdDevUncertainty(final_uncertainty),
                     mask=total_weight == 0, wcs=cubes[reference_cube_index].wcs)


@punch_task
def create_low_noise_task(
        cubes: list[PUNCHCube],
        nfi_cubes: list[PUNCHCube] | None = None,
        reference_time: str | datetime | None = None,
        exclude_outliers: bool = True,
        nfi_wfi_divide_radius: float | None = None,
        nfi_scale_factor: float = 1) -> PUNCHCube:
    """Create a low noise image from a set of inputs."""
    cube_count = len(cubes)
    cubes = [cube for cube in cubes if not (exclude_outliers and check_outlier(cube))]

    if isinstance(reference_time, str):
        reference_time = parse_datetime_str(reference_time)

    # TODO - Note to future self: be clever and use the outlier flag to excise bad data spatially.
    reference_cube_index = len(cubes)//2 - 1
    # Only the Level 3 polarized PTM -> PAM path uses propagated mean errors.
    if cubes and all(cube.meta.product_code == "PTM" and cube.meta["LEVEL"].value == "3" for cube in cubes):
        new_cube = merge_polarized_low_noise(cubes, reference_cube_index)
    else:
        new_cube = _merge_ndcubes(cubes, reference_cube_index=reference_cube_index)

    # TODO - will need to restore polarization resolution for starfield skipping

    new_code = cubes[0].meta.product_code[0] + "A" + cubes[0].meta.product_code[2]
    new_meta = NormalizedMetadata.load_template(new_code, "3")

    if nfi_wfi_divide_radius:
        nfi_wfi_mask = make_circular_mask(cubes[0].data.shape, nfi_wfi_divide_radius)
        if nfi_cubes:
            nfi_cubes = [cube for cube in nfi_cubes if not (exclude_outliers and check_outlier(cube))]
            nfi_data = np.array([cube.data for cube in nfi_cubes])
            # We'll want to figure out where there isn't good data in each NFI image. If an output pixel didn't have any
            # good samples to go on, we'll flag it in the output image's uncertainty. (This is mostly just handling the
            # occulter region.)
            maybe_masked = [(cube.data == 0) * np.isinf(cube.uncertainty.array) for cube in nfi_cubes]
            masked_pixels = np.all(maybe_masked, axis=0)

            median_nfi = nan_percentile(nfi_data, 50)
            new_cube.data[nfi_wfi_mask] = median_nfi[nfi_wfi_mask] * nfi_scale_factor

            # Uncertainty fill value for good samples
            new_cube.uncertainty.array[nfi_wfi_mask] = 1e-15
            # Set the uncertainty to inf elsewhere (e.g. in the occulter)
            new_cube.uncertainty.array[nfi_wfi_mask * masked_pixels] = np.inf

            new_meta["HAS_NFI4"] = 1
        # We've been given a radius for a NFI-only region, but we don't have NFI data. Make sure that region gets
        # cleared and marked properly.
        new_cube.data[:] = np.where(nfi_wfi_mask, 0, new_cube.data)
        new_cube.uncertainty.array[:] = np.where(
            nfi_wfi_mask, np.inf, new_cube.uncertainty.array)

    for k in cubes[0].meta.fits_keys:
        if k not in KEYWORD_OMIT and k in new_meta:
            new_meta[k] = cubes[reference_cube_index].meta[k].value

    # If any input data are excluded, flag this as an outlier
    if (cube_count != len(cubes)
            or not all(cube.meta["HAS_WFI1"] for cube in cubes)
            or not all(cube.meta["HAS_WFI2"] for cube in cubes)
            or not all(cube.meta["HAS_WFI3"] for cube in cubes)):
        new_meta["OUTLIER"] = 1

    new_meta.provenance = [c.meta["FILENAME"] for c in cubes]
    if nfi_cubes:
        new_meta.provenance += [c.meta["FILENAME"] for c in nfi_cubes]

    mean_date = average_datetime([cube.meta.datetime for cube in cubes])
    date_obs = mean_date if reference_time is None else reference_time
    try:
        times_beg = np.array([parse_datetime(cube.meta["DATE-BEG"].value).replace(tzinfo=UTC).timestamp()
                            for cube in cubes])
        times_end = np.array([parse_datetime(cube.meta["DATE-END"].value).replace(tzinfo=UTC).timestamp()
                            for cube in cubes])
    except ParserError:
        times_beg = np.array([parse_datetime(cube.meta["DATE-OBS"].value).replace(tzinfo=UTC).timestamp()
                            for cube in cubes])
        times_end = np.array([parse_datetime(cube.meta["DATE-OBS"].value).replace(tzinfo=UTC).timestamp()
                            for cube in cubes])

    new_meta["TYPECODE"] = new_code[0:2]
    new_meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    new_meta["DATE-OBS"] = date_obs.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    new_meta["DATE-AVG"] = mean_date.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    new_meta["DATE-BEG"] = datetime.fromtimestamp(np.min(times_beg),
                                                  tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    new_meta["DATE-END"] = datetime.fromtimestamp(np.max(times_end),
                                                  tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    new_cube.meta = new_meta

    return new_cube
