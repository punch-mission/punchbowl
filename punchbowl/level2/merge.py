
import numpy as np
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube
from prefect import task


@task
def merge_many_task(data: list[NDCube], trefoil_wcs: WCS) -> NDCube:
    """Merge many task and carefully combine uncertainties."""
    reprojected_data = np.stack([d.data for d in data], axis=-1)
    reprojected_uncertainty = np.stack([d.uncertainty.array for d in data], axis=-1)

    w = np.nansum(1/np.square(reprojected_data), axis=-1)
    trefoil_data = np.nansum(reprojected_data / np.square(reprojected_uncertainty), axis=2) / w
    trefoil_uncertainty = np.sqrt(1 / w)

    return NDCube(
        trefoil_data,
        uncertainty=StdDevUncertainty(trefoil_uncertainty),
        wcs=trefoil_wcs,
        meta=data[0].meta,  # TODO: how does the meta get merged?
    )
