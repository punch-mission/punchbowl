import astropy
import numpy as np
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube

from punchbowl.data import NormalizedMetadata
from punchbowl.prefect import punch_task
from punchbowl.util import average_datetime


@punch_task
def merge_many_polarized_task(data: list[NDCube | None], trefoil_wcs: WCS) -> NDCube:
    """Merge many task and carefully combine uncertainties."""
    trefoil_data_layers, trefoil_uncertainty_layers = [], []
    for polarization in [-60, 0, 60]:
        selected_images = [d for d in data if d is not None and d.meta["POLAR"].value == polarization]
        if len(selected_images) > 0:
            reprojected_data = np.stack([d.data for d in selected_images], axis=-1)
            reprojected_uncertainties = np.stack([d.uncertainty.array for d in selected_images], axis=-1)
            reprojected_uncertainties[reprojected_uncertainties <= 0] = np.inf
            reprojected_uncertainties[np.isinf(reprojected_uncertainties)] = 1E64
            reprojected_uncertainties[reprojected_data <= 0] = np.inf

            reprojected_weights = 1 / np.square(reprojected_uncertainties)

            trefoil_data_layers.append(np.nansum(reprojected_data * reprojected_weights, axis=2) /
                                       np.nansum(reprojected_weights, axis=2))
            trefoil_uncertainty_layers.append(1/np.nansum(np.sqrt(reprojected_weights), axis=2))
        else:
            trefoil_data_layers.append(np.zeros((4096, 4096)))
            trefoil_uncertainty_layers.append(np.zeros((4096, 4096))-999)

    output_meta = NormalizedMetadata.load_template("PTM", "2")
    output_dateobs = average_datetime([d.meta.datetime for d in data if d is not None]).isoformat()
    output_datebeg = min([d.meta.datetime for d in data if d is not None]).isoformat()
    output_dateend = max([d.meta.datetime for d in data if d is not None]).isoformat()
    output_meta["DATE-OBS"] = output_dateobs
    output_meta["DATE-BEG"] = output_datebeg
    output_meta["DATE-END"] = output_dateend
    trefoil_3d_wcs = astropy.wcs.utils.add_stokes_axis_to_wcs(trefoil_wcs, 2)
    return NDCube(
        data=np.stack(trefoil_data_layers, axis=0),
        uncertainty=StdDevUncertainty(np.stack(trefoil_uncertainty_layers, axis=0)),
        wcs=trefoil_3d_wcs,
        meta=output_meta,
    )

@punch_task
def merge_many_clear_task(data: list[NDCube | None], trefoil_wcs: WCS) -> NDCube:
    """Merge many task and carefully combine uncertainties."""
    trefoil_data_layers, trefoil_uncertainty_layers = [], []
    selected_images = [d for d in data if d is not None]

    if len(selected_images) > 0:
        reprojected_data = np.stack([d.data for d in selected_images], axis=-1)
        reprojected_uncertainties = np.stack([d.uncertainty.array for d in selected_images], axis=-1)
        reprojected_uncertainties[reprojected_uncertainties <= 0] = np.inf
        reprojected_uncertainties[np.isinf(reprojected_uncertainties)] = 1E64
        reprojected_uncertainties[reprojected_data <= 0] = np.inf

        reprojected_weights = 1/np.square(reprojected_uncertainties)

        trefoil_data_layers.append(np.nansum(reprojected_data * reprojected_weights, axis=-1) /
                                   np.nansum(reprojected_weights, axis=-1))
        trefoil_uncertainty_layers.append(1/np.sqrt(np.nansum(reprojected_weights, axis=-1)))
    else:
        trefoil_data_layers.append(np.zeros((4096, 4096)))
        trefoil_uncertainty_layers.append(np.zeros((4096, 4096))-999)

    output_meta = NormalizedMetadata.load_template("CTM", "Q")

    return NDCube(
        data=np.stack(trefoil_data_layers, axis=0).squeeze(),
        uncertainty=StdDevUncertainty(np.stack(trefoil_uncertainty_layers, axis=0).squeeze()),
        wcs=trefoil_wcs,
        meta=output_meta,
    )
