from datetime import UTC, datetime

import numpy as np
from astropy.nddata import StdDevUncertainty
from dateutil.parser import parse as parse_datetime
from ndcube import NDCube

from punchbowl.data.meta import NormalizedMetadata
from punchbowl.prefect import punch_task


@punch_task
def create_low_noise_task(cubes: list[NDCube]) -> NDCube:
    """Create a low noise image from a set of inputs."""
    cubes = [cube for cube in cubes if cube.meta["OUTLIER"].value != 1]

    ref_cube_index = len(cubes)//2 - 1

    data_stack = np.stack([cube.data for cube in cubes], axis=0)
    uncertainty_stack = np.array([cube.uncertainty.array for cube in cubes])

    uncertainty_stack[uncertainty_stack <= 0] = np.inf
    uncertainty_stack[~np.isfinite(uncertainty_stack)] = 1E64

    uncertainty_stack[np.isnan(data_stack)] = np.inf
    uncertainty_stack[data_stack == 0] = np.inf

    weight_stack = 1/np.square(uncertainty_stack)
    weight_stack[np.isnan(uncertainty_stack)] = np.nan

    new_data = np.nansum(data_stack * weight_stack, axis=0) / np.nansum(weight_stack, axis=0)

    final_uncertainty = np.sqrt(np.nansum(uncertainty_stack, axis=0)) / np.sqrt(new_data)

    new_code = cubes[0].meta.product_code[0] + "A" + cubes[0].meta.product_code[2]
    new_meta = NormalizedMetadata.load_template(new_code, "3")

    for k in cubes[0].meta.fits_keys:
        if k not in ("COMMENT", "HISTORY", "") and k in new_meta:
            new_meta[k] = cubes[ref_cube_index].meta[k].value

    times_obs = np.array([cube.meta.datetime.timestamp() for cube in cubes])
    times_beg = np.array([parse_datetime(cube.meta["DATE-BEG"].value).replace(tzinfo=UTC).timestamp()
                          for cube in cubes])
    times_end = np.array([parse_datetime(cube.meta["DATE-END"].value).replace(tzinfo=UTC).timestamp()
                          for cube in cubes])

    new_meta["TYPECODE"] = new_code[0:2]
    # TODO - distinction here? and mean versus midpoint.
    new_meta["DATE-OBS"] = datetime.fromtimestamp(np.mean(times_obs),
                                                  tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    new_meta["DATE-AVG"] = datetime.fromtimestamp(np.mean(times_obs),
                                                  tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    new_meta["DATE-BEG"] = datetime.fromtimestamp(np.min(times_beg),
                                                  tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    new_meta["DATE-END"] = datetime.fromtimestamp(np.max(times_end),
                                                  tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    return NDCube(data=new_data, uncertainty=StdDevUncertainty(final_uncertainty),
                  wcs=cubes[ref_cube_index].wcs, meta=new_meta)
