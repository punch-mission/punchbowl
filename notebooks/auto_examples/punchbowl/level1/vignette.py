import os
import pathlib
import warnings
from collections.abc import Callable

import numpy as np
from astropy.wcs import WCS
from ndcube import NDCube
from reproject import reproject_adaptive

from punchbowl.data import load_ndcube_from_fits
from punchbowl.exceptions import (
    IncorrectPolarizationStateWarning,
    IncorrectTelescopeWarning,
    InvalidDataError,
    LargeTimeDeltaWarning,
    NoCalibrationDataWarning,
)
from punchbowl.prefect import punch_task


@punch_task
def correct_vignetting_task(data_object: NDCube, vignetting_path: str | pathlib.Path | Callable | None) -> NDCube:
    """
    Prefect task to correct the vignetting of an image.

    Vignetting is a reduction of an image's brightness or saturation toward the
    periphery compared to the image center, created by the optical path. The
    Vignetting Module will transform the data through a flat-field correction
    map, to cancel out the effects of optical vignetting created by distortions
    in the optical path. This module also corrects detector gain variation and
    offset.

    Correction maps will be 2048*2048 arrays, to match the input data, and
    built using the starfield brightness pattern. Mathematical Operation:

        I'_{i,j} = I_i,j / FF_{i,j}

    Where I_{i,j} is the number of counts in pixel i, j. I'_{i,j} refers to the
    modified value. FF_{i,j} is the small-scale flat field factor for pixel i,
    j. The correction mapping will take into account the orientation of the
    spacecraft and its position in the orbit.

    Uncertainty across the image plane is calculated using the modelled
    flat-field correction with stim lamp calibration data. Deviations from the
    known flat-field are used to calculate the uncertainty in a given pixel.
    The uncertainty is convolved with the input uncertainty layer to produce
    the output uncertainty layer.


    Parameters
    ----------
    data_object : PUNCHData
        data on which to operate

    vignetting_path : pathlib
        path to vignetting function to apply to input data

    Returns
    -------
    PUNCHData
        modified version of the input with the vignetting corrected

    """
    if vignetting_path is None:
        data_object.meta.history.add_now("LEVEL1-correct_vignetting", "Vignetting skipped")
        msg=f"Calibration file {vignetting_path} is unavailable, vignetting correction not applied"
        warnings.warn(msg, NoCalibrationDataWarning)
    else:
        if isinstance(vignetting_path, Callable):
            vignetting_function, vignetting_path = vignetting_path()
        else:
            if isinstance(vignetting_path, str):
                vignetting_path = pathlib.Path(vignetting_path)
            if not vignetting_path.exists():
                msg = f"File {vignetting_path} does not exist."
                raise InvalidDataError(msg)
            vignetting_function = load_ndcube_from_fits(vignetting_path, include_provenance=False)
        vignetting_function_date = vignetting_function.meta.astropy_time
        observation_date = data_object.meta.astropy_time
        if abs((vignetting_function_date - observation_date).to("day").value) > 14:
            msg = f"Calibration file {vignetting_path} contains data created greater than 2 weeks from the obsveration"
            warnings.warn(msg, LargeTimeDeltaWarning)
        if vignetting_function.meta["TELESCOP"].value != data_object.meta["TELESCOP"].value:
            msg = f"Incorrect TELESCOP value within {vignetting_path}"
            warnings.warn(msg, IncorrectTelescopeWarning)
        if vignetting_function.meta["OBSLAYR1"].value != data_object.meta["OBSLAYR1"].value:
            msg = f"Incorrect polarization state within {vignetting_path}"
            warnings.warn(msg, IncorrectPolarizationStateWarning)
        if vignetting_function.data.shape != data_object.data.shape:
            msg = f"Incorrect vignetting function shape within {vignetting_path}"
            raise InvalidDataError(msg)

        data_object.data[:, :] /= vignetting_function.data[:, :]
        data_object.uncertainty.array[:, :] /= vignetting_function.data[:, :]
        data_object.meta.history.add_now("LEVEL1-correct_vignetting",
                                         f"Vignetting corrected using {os.path.basename(str(vignetting_path))}")
    return data_object


def generate_vignetting_calibration(path_vignetting: str, path_mask: str,
                                    spacecraft: str) -> np.ndarray:
    """Create calibration data for vignetting."""
    if spacecraft in ["1", "2", "3"]:
        with open(path_vignetting) as f:
            lines = f.readlines()

        with open(path_mask, "rb") as f:
            byte_array = f.read()
        mask = np.unpackbits(np.frombuffer(byte_array, dtype=np.uint8)).reshape(2048,2048)
        mask = mask.T

        num_bins, bin_size = lines[0].split()
        num_bins = int(num_bins)
        bin_size = int(bin_size)

        values = np.array([float(v) for line in lines[1:] for v in line.split()])
        vignetting_original = values[:num_bins**2].reshape((num_bins, num_bins))

        vignetting_original[np.isnan(vignetting_original)] = 0

        vignetting_original[6,:] = 0

        wcs_vignetting = WCS(naxis=2)

        wcs_wfi = WCS(naxis=2)
        wcs_wfi.wcs.cdelt = wcs_wfi.wcs.cdelt * vignetting_original.shape[0] / 2048.

        vignetting = reproject_adaptive((vignetting_original, wcs_vignetting),
                                        shape_out=(2048,2048),
                                        output_projection=wcs_wfi,
                                        boundary_mode="ignore")[0]

        vignetting = vignetting * mask

        vignetting[mask == 0] = 1

        return vignetting
    if spacecraft=="4":
        # TODO: implement NFI speckle inclusion
        return np.ones((2048,2048))
    raise RuntimeError(f"Unknown spacecraft {spacecraft}")
