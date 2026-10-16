from datetime import UTC, datetime

import astropy.units as u
import numpy as np
from astropy.nddata import StdDevUncertainty
from ndcube import NDCollection
from solpolpy import resolve

from punchbowl.data.meta import NormalizedMetadata, set_spacecraft_location_to_earth
from punchbowl.data.punchcube import PUNCHCube
from punchbowl.prefect import get_logger, punch_task


@punch_task
def convert_polarization(
        input_data: PUNCHCube) -> PUNCHCube:
    """
    Convert MZP to tB, pB and pB-prime, propagating errors only for Level 3 PIM inputs.

    Variances follow diag(A C A.T), with A_tBi = 2/3,
    A_pBi = -4/3 cos(2(theta_i - alpha)), and
    A_pBprime_i = -4/3 sin(2(theta_i - alpha)),
    C is diagonal with C_ii = sigma_i**2.
    The geometry is treated as fixed (C_ij=0 for i!=j);
    cross-channel covariance and shared calibration errors are not included.
    """
    logger = get_logger()
    logger.info("convert2tbpb started")
    is_pim = input_data.meta.product_code == "PIM" and input_data.meta["LEVEL"].value == "3"

    collection_contents = [(label,
                            PUNCHCube(data=input_data[i].data,
                                   wcs=input_data.wcs.dropaxis(2),
                                   meta={"POLAR": angle},
                                   mask=input_data[i].mask if is_pim else None))
                           for (label, i, angle) in zip(["M", "Z", "P"],
                                                        [0, 1, 2],
                                                        [-60 * u.deg, 0 * u.deg, 60 * u.deg], strict=False)]
    data_collection = NDCollection(collection_contents, aligned_axes="all")

    resolved_data_collection = resolve(data_collection, "bp3")

    new_data = np.stack([resolved_data_collection["tB"].data,
                                resolved_data_collection["pB"].data,
                                resolved_data_collection["pBp"].data], axis=0)
    new_uncertainty = None
    new_mask = resolved_data_collection["pB"].mask if is_pim else None
    if new_mask is not None:
        new_mask = np.broadcast_to(new_mask, new_data.shape).copy()
    if is_pim and input_data.uncertainty is not None:
        sigma = input_data.uncertainty.array
        valid = np.all(np.isfinite(input_data.data) & np.isfinite(sigma) & (sigma > 0), axis=0)
        if input_data.mask is not None:
            valid &= ~np.any(np.broadcast_to(input_data.mask, input_data.shape), axis=0)
        # Reuse the radial angle field returned with the resolved signal.
        alpha = resolved_data_collection["alpha"].data
        valid &= np.isfinite(alpha)
        variance = np.zeros(new_data.shape, dtype=float)
        for i, label in enumerate(["M", "Z", "P"]):
            theta = data_collection[label].meta["POLAR"].to_value(u.rad)
            phase = 2 * (theta - alpha)
            var_i = np.square(np.where(valid, sigma[i], 0).astype(float))
            variance[0] += (2 / 3)**2 * var_i
            variance[1] += (-4 / 3 * np.cos(phase))**2 * var_i
            variance[2] += (-4 / 3 * np.sin(phase))**2 * var_i
        new_mask = ~np.broadcast_to(valid, new_data.shape) | ~np.isfinite(new_data)
        variance[new_mask] = np.inf
        new_uncertainty = StdDevUncertainty(np.sqrt(variance))
    new_wcs = input_data.wcs.copy()

    output_meta = NormalizedMetadata.load_template("PTM", "3")
    output_meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    output_meta["DATE-AVG"] = input_data.meta["DATE-AVG"].value
    output_meta["DATE-OBS"] = input_data.meta["DATE-OBS"].value
    output_meta["DATE-BEG"] = input_data.meta["DATE-BEG"].value
    output_meta["DATE-END"] = input_data.meta["DATE-END"].value
    output = PUNCHCube(data=new_data, uncertainty=new_uncertainty,
                       mask=new_mask, wcs=new_wcs, meta=output_meta)
    output = set_spacecraft_location_to_earth(output)

    logger.info("convert2tbpb finished")
    output.meta.history.add_now("LEVEL3-convert2bp3", "Convert MZP to tB-pB-pBp")

    return output
