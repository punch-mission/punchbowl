import astropy
import numpy as np
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube

from punchbowl.data import NormalizedMetadata
from punchbowl.data.punch_io import encode_outliers
from punchbowl.prefect import punch_task
from punchbowl.util import average_datetime


def _merge_ndcubes(cubes: list[NDCube | None], reference_cube_index: int | None = None) -> NDCube:
    """Create a merged data product from a set of input data, weighting by uncertainty."""
    if cubes is not None:
        if reference_cube_index is None:
            reference_cube_index = len(cubes)//2 - 1

        data_stack = np.stack([cube.data for cube in cubes], axis=0)
        uncertainty_stack = np.array([cube.uncertainty.array for cube in cubes])

        uncertainty_stack[uncertainty_stack <= 0] = np.nan
        uncertainty_stack[~np.isfinite(uncertainty_stack)] = np.nan
        uncertainty_stack[data_stack == 0] = np.nan

        weight_stack = 1/np.square(uncertainty_stack)

        new_data = np.nansum(data_stack * weight_stack, axis=0) / np.nansum(weight_stack, axis=0)

        final_uncertainty = np.sqrt(np.sum(np.where(np.isfinite(uncertainty_stack), uncertainty_stack**2, 0), axis=0)) \
            / np.sqrt(np.sum(np.isfinite(uncertainty_stack), axis=0))

        final_uncertainty[final_uncertainty == 0] = np.inf
        final_uncertainty[~np.isfinite(final_uncertainty)] = np.inf

        return NDCube(data=new_data, uncertainty=StdDevUncertainty(final_uncertainty), \
                      wcs=cubes[reference_cube_index].wcs)
    return None


@punch_task
def merge_many_polarized_task(data: list[NDCube | None], trefoil_wcs: WCS, level: str = "2",
                              product_code: str = "PTM", maintain_nans: bool = False) -> NDCube:
    """Merge many task and carefully combine uncertainties."""
    data_layers, uncertainty_layers = [], []
    for polarization in [-60, 0, 60]:
        polar_data = [d for d in data if d is not None and hasattr(d, "meta") and d.meta["POLAR"].value == polarization]

        if len(polar_data) > 0:
            data_merged = _merge_ndcubes(polar_data)
            data_merged.meta = NormalizedMetadata.load_template(product_code, level=level)

            if maintain_nans:
                data_stack = np.stack([d.data for d in polar_data], axis=-1)
                was_nan = np.all(np.isnan(data_stack), axis=-1)
                data_merged.data[was_nan] = np.nan
        else:
            data_merged = NDCube(data = np.zeros((4096, 4096)),
                                uncertainty = StdDevUncertainty(np.full((4096, 4096), np.inf)),
                                wcs = trefoil_wcs,
                                meta = NormalizedMetadata.load_template(product_code, level=level))

        data_layers.append(data_merged.data)
        uncertainty_layers.append(data_merged.uncertainty.array)

    trefoil_3d_wcs = astropy.wcs.utils.add_stokes_axis_to_wcs(trefoil_wcs, 2)

    output_cube = NDCube(data=np.stack(data_layers, axis=0),
                         uncertainty=StdDevUncertainty(np.stack(uncertainty_layers, axis=0)),
                         wcs = trefoil_3d_wcs,
                         meta = NormalizedMetadata.load_template(product_code, level=level))

    output_cube.meta["OUTLIER"] = encode_outliers([d for d in data if d is not None])

    output_cube.meta["DATE-OBS"] = average_datetime([d.meta.datetime for d in data if d is not None],
                                                    ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    output_cube.meta["DATE-AVG"] = average_datetime([d.meta.datetime for d in data if d is not None],
                                                    ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    output_cube.meta["DATE-BEG"] = min([d.meta.datetime for d in data if d is not None],
                                       ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    output_cube.meta["DATE-END"] = max([d.meta.datetime for d in data if d is not None],
                                       ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    return output_cube


@punch_task
def merge_many_clear_task(
        data: list[NDCube | None], trefoil_wcs: WCS, level: str = "2", product_code: str = "CTM",
        maintain_nans: bool = False) -> NDCube:
    """Merge many task and carefully combine uncertainties."""
    if len(data) > 0:
        data = [d for d in data if d is not None]
        data_merged = _merge_ndcubes(data)
        data_merged.meta = NormalizedMetadata.load_template(product_code, level=level)

        if maintain_nans:
            data_stack = np.stack([d.data for d in data], axis=-1)
            was_nan = np.all(np.isnan(data_stack), axis=-1)
            data_merged.data[was_nan] = np.nan
    else:
        data_merged = NDCube(data = np.zeros((4096, 4096)),
                             uncertainty = StdDevUncertainty(np.full((4096, 4096), np.inf)),
                             wcs = trefoil_wcs,
                             meta = NormalizedMetadata.load_template(product_code, level=level))

    data_merged.meta["OUTLIER"] = encode_outliers([d for d in data if d is not None])

    data_merged.meta["DATE-OBS"] = average_datetime([d.meta.datetime for d in data if d is not None],
                                                    ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    data_merged.meta["DATE-AVG"] = average_datetime([d.meta.datetime for d in data if d is not None],
                                                    ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    data_merged.meta["DATE-BEG"] = min([d.meta.datetime for d in data if d is not None],
                                       ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    data_merged.meta["DATE-END"] = max([d.meta.datetime for d in data if d is not None],
                                       ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    return data_merged
