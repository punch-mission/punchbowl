import os
import pathlib
import warnings
from pathlib import Path
from datetime import UTC, datetime

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from ndcube import NDCube
from reproject import reproject_adaptive
from scipy.ndimage import binary_dilation, binary_erosion, grey_closing

from punchbowl.data import load_ndcube_from_fits
from punchbowl.data.meta import NormalizedMetadata
from punchbowl.data.punch_io import get_base_file_name
from punchbowl.exceptions import (
    IncorrectPolarizationStateWarning,
    IncorrectTelescopeWarning,
    InvalidDataError,
    LargeTimeDeltaWarning,
    NoCalibrationDataWarning,
)
from punchbowl.level1.sqrt import decode_sqrt_data
from punchbowl.prefect import punch_task
from punchbowl.util import DataLoader, load_spacecraft_mask


@punch_task
def correct_vignetting_task(data_object: NDCube, vignetting_path: str | pathlib.Path | DataLoader | None) -> NDCube:
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
        if isinstance(vignetting_path, DataLoader):
            vignetting_function = vignetting_path.load()
            vignetting_path = vignetting_path.src_repr()
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
            msg = f"Calibration file {vignetting_path} contains data created greater than 2 weeks from the observation"
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


def generate_vignetting_calibration_wfi(path_vignetting: str,
                                        path_mask: str,
                                        spacecraft: str,
                                        vignetting_threshold: float = 1.2,
                                        rows_ignore: tuple = (13,15),
                                        rows_adjust: tuple = (15,16),
                                        rows_adjust_source: tuple = (16,20),
                                        mask_erosion: tuple = (6,6)) -> np.ndarray:
    """
    Create calibration data for vignetting.

    Parameters
    ----------
    path_vignetting : str
        path to raw input vignetting function
    path_mask : str
        path to spacecraft mask function
    spacecraft : str
        spacecraft number
    vignetting_threshold : float, optional
        threshold for bad vignetting pixels, by default 1.2
    rows_ignore : tuple, optional
        rows to exclude entirely from original vignetting data, by default (13,15) for 128x128 input
    rows_adjust : tuple, optional
        rows to adjust to the minimum of a set of rows above (per column), by default (15,16) for 128x128 input
    rows_adjust_source : tuple, optional
        rows to use for statistics to adjust vignetting rows as above, by default (16,20) for 128x128 input
    mask_erosion: tuple, optional
        kernel to use in erosion operation to reduce the mask applied to the vignetting function, by default (6,6)

    Returns
    -------
    np.ndarray
        vignetting function array

    """
    if spacecraft in ["1", "2", "3"]:
        if not os.path.exists(path_vignetting):
            return np.ones((2048,2048))

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
        vignetting = values[:num_bins**2].reshape((num_bins, num_bins))

        vignetting[vignetting > vignetting_threshold] = np.nan

        vignetting[rows_ignore[0]:rows_ignore[1],:] = np.nan
        vignetting[rows_adjust[0]:rows_adjust[1],:] = np.min(vignetting[rows_adjust_source[0]:rows_adjust_source[1],:],
                                                             axis=0)

        wcs_vignetting = WCS(naxis=2)

        wcs_wfi = WCS(naxis=2)
        wcs_wfi.wcs.cdelt = wcs_wfi.wcs.cdelt * vignetting.shape[0] / 2048.

        vignetting_reprojected = reproject_adaptive((vignetting, wcs_vignetting),
                                                shape_out=(2048,2048),
                                                output_projection=wcs_wfi,
                                                boundary_mode="ignore",
                                                bad_value_mode="ignore",
                                                return_footprint=False)

        mask = binary_erosion(mask, structure=np.ones(mask_erosion))

        vignetting_reprojected = vignetting_reprojected * mask

        vignetting_reprojected[mask == 0] = 1

        return vignetting_reprojected
    if spacecraft=="4":
        raise RuntimeError("Please use the NFI vignetting generator function.")
    raise RuntimeError(f"Unknown spacecraft {spacecraft}")


def generate_vignetting_calibration_nfi(input_files: list[str], # noqa: C901
                                        path_speckle: str,
                                        path_mask: str,
                                        path_dark: str,
                                        polarizer: str,
                                        dateobs: str,
                                        version: str,
                                        output_path: str | None = None,
                                        max_files: int = -1) -> np.ndarray | str:
    """
    Create calibration data for vignetting for the NFI spacecraft.

    Parameters
    ----------
    input_files : list[str]
        Paths to input NFI files for processing
    path_speckle : str
        Path to the speckle mask FITS file
    path_mask : str
        Path to the NFI mask bin file
    path_dark : str
        Path to the dark frame FITS file
    polarizer : str
        Polarizer name
    dateobs : str
        Timestamp for calibration file
    version : str
        File version
    output_path : str | None
        Path to calibration file output
    max_files : int
        If set, only the first this many non-outlier files will be loaded and used


    Returns
    -------
    np.ndarray | str
        vignetting function array or written file path

    """
    if input_files is None:
        return np.ones((2048,2048))

    # Load speckle mask and dark frame
    with fits.open(path_speckle) as hdul:
        specklemask = np.fliplr(hdul[0].data)

    with fits.open(path_dark) as hdul:
        nfidark = hdul[1].data

    # Load and square root decode input data
    cubes = []
    for file in input_files:
        try:
            cube = load_ndcube_from_fits(file)
        except: # noqa: E722
            print(f"Error loading {file}") # noqa: T201
            continue
        if cube.meta["OFFSET"].value is None:
            cube.meta["OFFSET"] = 400
        # Reject outlier images in vignetting calculation if they have been flagged
        if "OUTLIER" in cube.meta:
            if cube.meta["OUTLIER"].value == 0:
                cubes.append(decode_sqrt_data.fn(cube))
        # If outlier rejection was not applied, manually reject outliers using these human-derived checks
        else:
            if 490 <= cube.meta["DATAMDN"].value <= 655 and cube.meta["DATAP99"].value != 4095:
                cubes.append(decode_sqrt_data.fn(cube))
        if max_files > 0 and len(cubes) >= max_files:
            break

    # Subtract dark frame
    for cube in cubes:
        cube.data[...] -= nfidark

    # Build speckle boundary mask
    inverted_mask = 1 - specklemask
    dilated = binary_dilation(inverted_mask, structure=np.ones((3, 3)))
    boundary_mask = dilated & (~inverted_mask)

    # Stack image data
    images = np.array([cube.data for cube in cubes])

    # If only one file exists in the datacube, do not create the vignetting correction
    if len(images.shape) != 3:
        return None

    applied_images = images * boundary_mask
    applied_speck = images * specklemask

    # Compute averages and construct flatfield
    avg_images = np.nanmean(applied_images, axis=0)
    avg_img_darkremoved = np.nanmean(images, axis=0)
    avg_speck = np.nanmean(applied_speck, axis=0)
    avg_speckfilled = grey_closing(avg_images, structure=np.ones((7, 7)))

    nficlean = avg_speckfilled * inverted_mask + avg_speck
    nfiflat = avg_img_darkremoved / nficlean

    # Load spacecraft mask
    mask_nfi = load_spacecraft_mask(path_mask)

    # Deal with infs and remask
    nfiflat[np.isinf(nfiflat)] = 1.
    nfiflat[mask_nfi == 0] = 1

    # Generate an output metadata and NDCube
    m = NormalizedMetadata.load_template(f"G{polarizer}4", "1")
    m["DATE-OBS"] = dateobs
    m["FILEVRSN"] = version
    m["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    cube = NDCube(data=nfiflat.astype("float32"), wcs=cube.wcs, meta=m)

    if output_path is not None:
        filename = Path(output_path) / f"{get_base_file_name(cube)}.fits"

        full_header = cube.meta.to_fits_header(wcs=cube.wcs)
        full_header["FILENAME"] = os.path.basename(filename)

        hdu_data = fits.ImageHDU(data=cube.data,
                                     header=full_header,
                                     name="Primary data array")
        hdu_provenance = fits.BinTableHDU.from_columns(fits.ColDefs([fits.Column(
            name="provenance", format="A40", array=np.char.array(cube.meta.provenance))]))
        hdu_provenance.name = "File provenance"

        hdul = cube.wcs.to_fits()
        hdul[0] = fits.PrimaryHDU()
        hdul.insert(1, hdu_data)

        hdul.append(hdu_provenance)
        hdul.writeto(filename, overwrite=True, checksum=True)
        hdul.close()

        return filename
    return cube.data
