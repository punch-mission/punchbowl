# Core Python imports
from typing import List, Tuple

# Third party imports
import numpy as np
import reproject
from astropy.wcs import WCS
from prefect import flow, get_run_logger, task

# Punchbowl imports
from punchbowl.data import PUNCHData


@task
def merge_many_task(data: List[PUNCHData], trefoil_wcs: WCS) -> PUNCHData:
    # TODO: add docstring
    reprojected_data = np.stack([d.data for d in data], axis=-1)
    reprojected_uncertainty = np.stack([d.uncertainty.array for d in data], axis=-1)

    # Merge these data
    # Carefully deal with missing data (NaN) by only ignoring a pixel missing from all observations
    trefoil_data = np.nansum(reprojected_data * reprojected_uncertainty, axis=2) / np.nansum(reprojected_uncertainty,
                                                                                             axis=2)
    trefoil_uncertainty = np.amax(reprojected_uncertainty)

    # Pack up an output data object
    data_object = PUNCHData(trefoil_data, uncertainty=trefoil_uncertainty, wcs=trefoil_wcs,
                            meta=data[0].meta)
    # TODO: what do we do with the meta data???? shouldn't it merge

    return data_object
