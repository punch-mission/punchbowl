from __future__ import annotations

import os.path

import astropy.units as u
import astropy.wcs.wcsapi
import matplotlib as mpl
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.fits import Header
from astropy.nddata import StdDevUncertainty
from astropy.time import Time
from astropy.wcs import WCS
from ndcube import NDCube
from sunpy.coordinates import frames, sun
from sunpy.map import solar_angular_radius

from punchbowl.data.meta import NormalizedMetadata
from punchbowl.data.wcs import calculate_helio_wcs_from_celestial

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_base_file_name(cube: NDCube) -> str:
    """Determine the base file name without file type extension."""
    obscode = cube.meta["OBSCODE"].value
    file_level = cube.meta["LEVEL"].value
    type_code = cube.meta["TYPECODE"].value
    date_string = cube.meta.datetime.strftime("%Y%m%d%H%M%S")
    file_version = cube.meta["FILEVRSN"].value
    return "PUNCH_L" + file_level + "_" + type_code + obscode + "_" + date_string + "_v" + file_version


def write_ndcube_to_fits(cube: NDCube,
                         filename: str,
                         overwrite: bool = True) -> None:
    """Write an NDCube as a FITS file."""
    if filename.endswith(".fits"):
        _write_fits(cube, filename, overwrite=overwrite)
    else:
        msg = (
            "Filename must have a valid file extension `.fits`"
            f"Found: {os.path.splitext(filename)[1]}"
        )
        raise ValueError(
            msg,
        )


def construct_wcs_header_fields(cube: NDCube) -> Header:
    """
    Compute primary and secondary WCS header cards to add to a data object.

    Returns
    -------
    Header

    """
    date_obs = Time(cube.meta.datetime)

    celestial_wcs_header = cube.wcs.to_header()
    output_header = astropy.io.fits.Header()

    unused_keys = [
        "DATE-OBS",
        "DATE-BEG",
        "DATE-AVG",
        "DATE-END",
        "DATE",
        "MJD-OBS",
        "TELAPSE",
        "RSUN_REF",
        "TIMESYS",
    ]

    helio_wcs, p_angle = calculate_helio_wcs_from_celestial(
        wcs_celestial=cube.wcs, date_obs=date_obs, data_shape=cube.data.shape,
    )

    helio_wcs_hdul_reference = helio_wcs.to_fits()
    helio_wcs_header = helio_wcs_hdul_reference[0].header

    for key in unused_keys:
        if key in celestial_wcs_header:
            del celestial_wcs_header[key]
        if key in helio_wcs_header:
            del helio_wcs_header[key]

    if cube.meta["CTYPE1"] is not None:
        for key, value in helio_wcs.to_header().items():
            output_header[key] = value
    if cube.meta["CTYPE1A"] is not None:
        for key, value in celestial_wcs_header.items():
            output_header[key + "A"] = value

    center_helio_coord = SkyCoord(
        helio_wcs.wcs.crval[0] * u.deg,
        helio_wcs.wcs.crval[1] * u.deg,
        frame=frames.Helioprojective,
        obstime=date_obs,
        observer="earth",
    )

    output_header["RSUN_ARC"] = solar_angular_radius(center_helio_coord).value
    output_header["SOLAR_EP"] = p_angle.value
    output_header["CAR_ROT"] = float(sun.carrington_rotation_number(t=date_obs))

    return output_header


def _write_fits(cube: NDCube, filename: str, overwrite: bool = True, uncertainty_quantize_level: float = -2.0) -> None:
    _update_statistics(cube)

    hdul = cube.wcs.to_fits()
    full_header = cube.meta.to_fits_header(wcs=cube.wcs)

    hdu_data = fits.CompImageHDU(data=cube.data,
                                 header=full_header,
                                 name="Primary data array")
    hdu_uncertainty = fits.CompImageHDU(data=_pack_uncertainty(cube),
                                        header=full_header,
                                        name="Uncertainty array",
                                        quantize_level=uncertainty_quantize_level)

    hdul[0] = fits.PrimaryHDU()
    hdul.insert(1, hdu_data)
    hdul.insert(2, hdu_uncertainty)

    hdul.writeto(filename, overwrite=overwrite, checksum=True)


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

    nonzero_data = cube.data[np.where(cube.data != 0)].flatten()

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
        hdu_index = next((i for i, hdu in enumerate(hdul) if hdu.data is not None), 0)
        primary_hdu = hdul[hdu_index]
        data = primary_hdu.data
        header = primary_hdu.header
        # Reset checksum and datasum to match astropy.io.fits behavior
        header["CHECKSUM"] = ""
        header["DATASUM"] = ""
        meta = NormalizedMetadata.from_fits_header(header)
        wcs = WCS(header, hdul, key=key)
        unit = u.ct

        if len(hdul) > hdu_index + 1:
            secondary_hdu = hdul[hdu_index+1]
            uncertainty = _unpack_uncertainty(secondary_hdu.data, data)
            uncertainty = StdDevUncertainty(uncertainty)
        else:
            uncertainty = None

    return NDCube(
        data.newbyteorder().byteswap(inplace=False),
        wcs=wcs,
        uncertainty=uncertainty,
        meta=meta,
        unit=unit,
    )
