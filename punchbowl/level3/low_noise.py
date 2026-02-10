from datetime import UTC, datetime

import astropy.units as u
import numpy as np
import solpolpy
from astropy.nddata import StdDevUncertainty
from dateutil.parser import parse as parse_datetime
from dateutil.parser import parse as parse_datetime_str
from ndcube import NDCollection, NDCube

from punchbowl.data.meta import NormalizedMetadata
from punchbowl.data.punch_io import check_outlier
from punchbowl.level2.merge import _merge_ndcubes
from punchbowl.prefect import punch_task
from punchbowl.util import average_datetime

KEYWORD_OMIT = ("COMMENT", "HISTORY", "", "NAXIS3", "OBSTYPE", "OBS-MODE", "OBSLAYR1", "OBSLAYR2", "OBSLAYR3")

@punch_task
def create_low_noise_task(cubes: list[NDCube], reference_time: str | datetime | None = None) -> NDCube:
    """Create a low noise image from a set of inputs."""
    cube_count = len(cubes)
    cubes = [cube for cube in cubes if not check_outlier(cube)]

    if isinstance(reference_time, str):
        reference_time = parse_datetime_str(reference_time)

    # TODO - Note to future self: be clever and use the outlier flag to excise bad data spatially.
    reference_cube_index = len(cubes)//2 - 1
    new_cube = _merge_ndcubes(cubes, reference_cube_index=reference_cube_index)

    if new_cube.data.ndim == 3:
        mzp_map = [-60, 0, 60]
        layer_map = {"M": 0, "Z": 1, "P": 2}

        mzp_collection = NDCollection(
            [(k, NDCube(
                data=new_cube.data[layer_map[k], ...],
                wcs=new_cube.wcs[layer_map[k]],
                meta={
                    "POLAR": mzp_map[layer_map[k]] * u.degree,
                    "POLAROFF": 0,
                    "POLARREF": "solar",
                },
            ))
            for k in ["M", "Z", "P"]],
            aligned_axes=(0, 1),
        )

        bpb_collection = solpolpy.resolve(mzp_collection, "bpb")

    new_code = cubes[0].meta.product_code[0] + "A" + cubes[0].meta.product_code[2]
    new_meta = NormalizedMetadata.load_template(new_code, "3")

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

    mean_date = average_datetime([cube.meta.datetime for cube in cubes])
    date_obs = mean_date if reference_time is None else reference_time
    times_beg = np.array([parse_datetime(cube.meta["DATE-BEG"].value).replace(tzinfo=UTC).timestamp()
                          for cube in cubes])
    times_end = np.array([parse_datetime(cube.meta["DATE-END"].value).replace(tzinfo=UTC).timestamp()
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

    if new_cube.data.ndim == 3:
        # # TODO - Fully propagate uncertainty
        new_uncertainty = np.copy(new_cube.uncertainty.array[0,...])
        new_uncertainty[np.isfinite(new_uncertainty)] = 0
        new_uncertainty = np.stack([new_uncertainty, new_uncertainty], axis=0)
        return NDCube(data = np.stack([bpb_collection[k].data for k in ["B", "pB"]]),
                           uncertainty=StdDevUncertainty(new_uncertainty),
                           wcs=new_cube.wcs,
                           meta=new_cube.meta)

    return new_cube
