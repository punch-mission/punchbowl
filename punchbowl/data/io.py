from __future__ import annotations

import hashlib
import os.path

import astropy.units as u
import matplotlib as mpl
import numpy as np
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube

from punchbowl.data.meta import NormalizedMetadata

_ROOT = os.path.abspath(os.path.dirname(__file__))

def write_file_hash(path: str) -> None:
    """Create a SHA-256 hash for a file."""
    file_hash = hashlib.sha256()
    with open(path, "rb") as f:
        fb = f.read()
        file_hash.update(fb)

    with open(path + ".sha", "w") as f:
        f.write(file_hash.hexdigest())


def get_base_file_name(cube: NDCube) -> str:
    """Determine the base file name without file type extension."""
    obscode = cube.meta["OBSCODE"].value
    file_level = cube.meta["LEVEL"].value
    type_code = cube.meta["TYPECODE"].value
    date_string = cube.meta.datetime.strftime("%Y%m%d%H%M%S")
    file_version = cube.meta["FILEVRSN"].value
    file_version = "1" if file_version == "" else file_version  # file version should never be empty!
    return "PUNCH_L" + file_level + "_" + type_code + obscode + "_" + date_string + "_v" + file_version


def write_ndcube_to_fits(cube: NDCube,
                         filename: str,
                         overwrite: bool = True,
                         write_hash: bool = True,
                         uncertainty_quantize_level: float = 16) -> None:
    """Write an NDCube as a FITS file."""
    if filename.endswith(".fits"):
        _update_statistics(cube)

        full_header = cube.meta.to_fits_header(wcs=cube.wcs)

        hdu_data = fits.CompImageHDU(data=cube.data,
                                     header=full_header,
                                     name="Primary data array")
        hdu_uncertainty = fits.CompImageHDU(data=_pack_uncertainty(cube),
                                            header=full_header,
                                            name="Uncertainty array",
                                            quantize_level=uncertainty_quantize_level)

        hdul = cube.wcs.to_fits()
        hdul[0] = fits.PrimaryHDU()
        hdul.insert(1, hdu_data)
        hdul.insert(2, hdu_uncertainty)
        hdul.writeto(filename, overwrite=overwrite, checksum=True)
        hdul.close()
        if write_hash:
            write_file_hash(filename)
    else:
        msg = (
            "Filename must have a valid file extension `.fits`"
            f"Found: {os.path.splitext(filename)[1]}"
        )
        raise ValueError(
            msg,
        )


def _pack_uncertainty(cube: NDCube) -> np.ndarray:
    """Compress the uncertainty for writing to file."""
    return np.zeros_like(cube.data) - 999 if cube.uncertainty is None else 1 / (cube.uncertainty.array / cube.data)


def _unpack_uncertainty(uncertainty_array: np.ndarray, data_array: np.ndarray) -> np.ndarray:
    """Uncompress the uncertainty when reading from a file."""
    return (1/uncertainty_array) * data_array


def _write_ql(cube: NDCube, filename: str, overwrite: bool = True) -> None:
    if os.path.isfile(filename) and not overwrite:
        msg = f"File {filename} already exists. If you mean to replace it then use the argument 'overwrite=True'."
        raise OSError(
            msg,
        )

    if cube.data.ndim != 2:
        msg = "Specified output data should have two-dimensions."
        raise ValueError(msg)

    # Scale data array to 8-bit values
    output_data = int(np.fix(np.interp(cube.data, (cube.data.min(), cube.data.max()), (0, 2**8 - 1))))

    # Write image to file
    mpl.image.saveim(filename, output_data)


def _update_statistics(cube: NDCube) -> None:
    """Update image statistics in metadata before writing to file."""
    # TODO - Determine DSATVAL omniheader value in calibrated units for L1+

    cube.meta["DATAZER"] = len(np.where(cube.data == 0)[0])

    cube.meta["DATASAT"] = len(np.where(cube.data >= cube.meta["DSATVAL"].value)[0])

    cube.data[np.logical_not(np.isfinite(cube.data))] = 0
    nonzero_data = cube.data[np.isfinite(cube.data) * (cube.data != 0)].flatten()

    if len(nonzero_data) > 0:
        cube.meta["DATAAVG"] = np.nanmean(nonzero_data).item()
        cube.meta["DATAMDN"] = np.nanmedian(nonzero_data).item()
        cube.meta["DATASIG"] = np.nanstd(nonzero_data).item()
    else:
        cube.meta["DATAAVG"] = -999.0
        cube.meta["DATAMDN"] = -999.0
        cube.meta["DATASIG"] = -999.0

    percentile_percentages = [1, 10, 25, 50, 75, 90, 95, 98, 99]
    percentile_values = np.nanpercentile(nonzero_data, percentile_percentages)
    if np.any(np.isnan(percentile_values)):  # report nan if any of the values are nan
        percentile_values = [-999.0 for _ in percentile_percentages]

    for percent, value in zip(percentile_percentages, percentile_values, strict=False):
        cube.meta[f"DATAP{percent:02d}"] = value

    cube.meta["DATAMIN"] = float(np.nanmin(cube.data))
    cube.meta["DATAMAX"] = float(np.nanmax(cube.data))


def load_ndcube_from_fits(path: str, key: str = " ") -> NDCube:
    """Load an NDCube from a FITS file."""
    with fits.open(path) as hdul:
        primary_hdu = hdul[1]
        data = primary_hdu.data
        header = primary_hdu.header
        # Reset checksum and datasum to match astropy.io.fits behavior
        header["CHECKSUM"] = ""
        header["DATASUM"] = ""
        meta = NormalizedMetadata.from_fits_header(header)
        wcs = WCS(header, hdul, key=key)
        unit = u.ct

        secondary_hdu = hdul[2]
        uncertainty = _unpack_uncertainty(secondary_hdu.data.astype(float), data)
        uncertainty = StdDevUncertainty(uncertainty)

    return NDCube(
        data.view(dtype=data.dtype.newbyteorder()).byteswap().astype(float),
        wcs=wcs,
        uncertainty=uncertainty,
        meta=meta,
        unit=unit,
    )
