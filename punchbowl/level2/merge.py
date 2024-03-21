# Core Python imports
from typing import List

# Third party imports
import numpy as np
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from prefect import task

# Punchbowl imports
from punchbowl.data import PUNCHData


@task
def merge_many_task(data: List[PUNCHData], trefoil_wcs: WCS) -> PUNCHData:
    # TODO: add docstring
    reprojected_data = np.stack([d.data for d in data], axis=-1)
    reprojected_uncertainty = np.stack([d.uncertainty.array for d in data], axis=-1)

    # Merge these data
    # Carefully deal with missing data (NaN) by only ignoring a pixel missing from all observations
    trefoil_data = np.nansum(reprojected_data * reprojected_uncertainty, axis=2) / np.nansum(
        reprojected_uncertainty, axis=2
    )
    trefoil_uncertainty = np.amax(reprojected_uncertainty, axis=-1)

    # Pack up an output data object
    data_object = PUNCHData(
        trefoil_data, uncertainty=StdDevUncertainty(trefoil_uncertainty), wcs=trefoil_wcs, meta=data[0].meta
    )
    data_object.meta["LEVEL"] = "2"
    # TODO: what do we do with the meta data???? shouldn't it merge

    return data_object
