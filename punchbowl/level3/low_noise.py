from datetime import UTC, datetime

import numpy as np
from dateutil.parser import parse as parse_datetime
from ndcube import NDCube

from punchbowl.data.meta import NormalizedMetadata
from punchbowl.data.punch_io import check_outlier
from punchbowl.level2.merge import _merge_ndcubes
from punchbowl.prefect import punch_task


@punch_task
def create_low_noise_task(cubes: list[NDCube]) -> NDCube:
    """Create a low noise image from a set of inputs."""
    cubes = [cube for cube in cubes if not check_outlier(cube)]

    reference_cube_index = len(cubes)//2 - 1
    new_cube = _merge_ndcubes(cubes, reference_cube_index=reference_cube_index)

    new_code = cubes[0].meta.product_code[0] + "A" + cubes[0].meta.product_code[2]
    new_meta = NormalizedMetadata.load_template(new_code, "3")

    for k in cubes[0].meta.fits_keys:
        if k not in ("COMMENT", "HISTORY", "") and k in new_meta:
            new_meta[k] = cubes[reference_cube_index].meta[k].value

    times_obs = np.array([cube.meta.datetime.timestamp() for cube in cubes])
    times_beg = np.array([parse_datetime(cube.meta["DATE-BEG"].value).replace(tzinfo=UTC).timestamp()
                          for cube in cubes])
    times_end = np.array([parse_datetime(cube.meta["DATE-END"].value).replace(tzinfo=UTC).timestamp()
                          for cube in cubes])

    new_meta["TYPECODE"] = new_code[0:2]
    new_meta["DATE-OBS"] = datetime.fromtimestamp(np.mean(times_obs),
                                                  tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    new_meta["DATE-AVG"] = datetime.fromtimestamp(np.mean(times_obs),
                                                  tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    new_meta["DATE-BEG"] = datetime.fromtimestamp(np.min(times_beg),
                                                  tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    new_meta["DATE-END"] = datetime.fromtimestamp(np.max(times_end),
                                                  tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    new_cube.meta = new_meta

    return new_cube
