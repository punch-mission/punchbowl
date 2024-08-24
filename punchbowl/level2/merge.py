
import numpy as np
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube
from prefect import task

from punchbowl.data import NormalizedMetadata


@task
def merge_many_task(data: list[NDCube], trefoil_wcs: WCS) -> NDCube:
    """Merge many task and carefully combine uncertainties."""
    trefoil_data_layers, trefoil_uncertainty_layers = [], []
    for polarization in [-60, 0, 60]:
        reprojected_data = np.stack([d.data for d in data if d.meta["POLAR"].value == polarization], axis=-1)
        reprojected_weights = np.stack([1/np.square(d.uncertainty.array) for d in data if d.meta["POLAR"].value == polarization],
                                           axis=-1)
        reprojected_weights[reprojected_weights <= 0] = 1E-8
        # mask = np.zeros_like(reprojected_weights, dtype=bool)
        # mask[5:2043, 5:2043, :] = True
        # mask = np.logical_not(mask)
        # reprojected_weights[mask] = 1E-8

        trefoil_data_layers.append(np.nansum(reprojected_data * reprojected_weights, axis=2) / np.nansum(reprojected_weights, axis=2))
        trefoil_uncertainty_layers.append(1/np.nansum(np.sqrt(reprojected_weights), axis=2))

    output_meta = NormalizedMetadata.load_template("PTM", "2")
    output_meta['DATE-OBS'] = data[0].meta['DATE-OBS'].value  # TODO: do this better and fill rest of meta

    return NDCube(
        data=np.stack(trefoil_data_layers, axis=0),
        uncertainty=StdDevUncertainty(np.stack(trefoil_uncertainty_layers, axis=0)),
        wcs=trefoil_wcs,
        meta=output_meta
    )
