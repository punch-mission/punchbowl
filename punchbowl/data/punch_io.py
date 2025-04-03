from __future__ import annotations

import hashlib
import os.path
import subprocess
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from matplotlib.colors import LogNorm
from ndcube import NDCube
from PIL import Image, ImageDraw, ImageFont

from punchbowl.data.meta import NormalizedMetadata
from punchbowl.data.visualize import cmap_punch

_ROOT = os.path.abspath(os.path.dirname(__file__))

CALIBRATION_ANNOTATION = "{OBSRVTRY} - {TYPECODE}{OBSCODE} - {DATE-OBS} - exptime: {EXPTIME} s - polarizer: {POLAR} deg"


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


def write_ndcube_to_quicklook(cube: NDCube,
                              filename: str,
                              layer: int | None = None,
                              vmin: float = 1e-15,
                              vmax: float = 8e-13,
                              annotation: str | None = None) -> None:
    """
    Write an NDCube to a Quicklook format as a jpeg.

    Parameters
    ----------
    cube : NDCube
        data cube to visualize
    filename : str
        path to save output, must end in .jp2, .j2k, .jpeg, .jpg
    layer : int | None
        if the cube is 3D, then selects cube.data[layer] for visualization
    vmin : float
        the lower limit value to visualize
    vmax : float
        the upper limit value to visualize
    annotation : str | None
        a formatted string to add to the bottom of the image as a label
        can access metadata by key, e.g. "typecode={TYPECODE}" would write the data's typecode into the image

    Returns
    -------
    None

    """
    if (len(cube.data.shape) != 2) and layer is None:
        msg = "Output data must be two-dimensional, or a layer must be specified"
        raise ValueError(msg)

    if not filename.endswith((".jp2", ".j2k", ".jpeg", ".jpg")):
        msg = ("Filename must have a valid file extension `.jpeg`, `jpg`, `.jp2` or `.j2k`"
               f"Found: {os.path.splitext(filename)[1]}")
        raise ValueError(msg)

    norm = LogNorm(vmin=vmin, vmax=vmax)

    if layer is not None:  # noqa: SIM108
        image = cube.data[layer, :, :]
    else:
        image = cube.data

    scaled_arr = (cmap_punch(norm(np.flipud(image))) * 255).astype(np.uint8)

    pil_image = Image.fromarray(scaled_arr, mode="RGBA")

    if annotation:
        pad_height = int(image.shape[1] * 50 / 2048)
        padded_image = Image.new("RGB", (pil_image.width, pil_image.height + pad_height))

        padded_image.paste(pil_image, (0, 0))  # copies the original image data into the padded image

        draw = ImageDraw.Draw(padded_image)
        font = ImageFont.load_default(size=int(pad_height / 2))

        text = annotation.format(**cube.meta)
        text_offset = int(10 * image.shape[1] / 2048)
        text_position = (text_offset, pil_image.height + text_offset)
        draw.text(text_position, text, font=font, fill=(255, 255, 255))
        pil_image = padded_image

    pil_image.save(filename)


def write_quicklook_to_mp4(files: list[str],
                           filename: str,
                           framerate: int = 5,
                           resolution: int = 1024,
                           codec: str = "libx264") -> None:
    """
    Write a list of input quicklook jpeg2000 files to an output mp4 animation.

    Parameters
    ----------
    files : list[str]
        List of input files to animate
    filename : str
        Output filename
    framerate : int, optional
        Frame rate (default 5)
    resolution : int, optional
        Output resolution (default 1024)
    codec : str, optional
        Codec to use for encoding. For GPU acceleration.
        "h264_videotoolbox" can be used on ARM Macs, "h264_nvenc" can be used on Intel machines.

    """
    input_sequence = f"concat:{'|'.join(files)}"

    ffmpeg_command = [
        "ffmpeg",
        "-framerate", str(framerate),
        "-i", input_sequence,
        "-vf", f"scale=-1:{resolution}",
        "-c:v", codec,
        "-pix_fmt", "yuv420p",
        "-y",
        filename,
    ]

    subprocess.run(ffmpeg_command, check=False)  # noqa: S603


def write_ndcube_to_fits(cube: NDCube,
                         filename: str,
                         overwrite: bool = False,
                         write_hash: bool = True,
                         skip_stats: bool = False,
                         uncertainty_quantize_level: float = 16) -> None:
    """Write an NDCube as a FITS file."""
    if filename.endswith(".fits"):
        if not skip_stats:
            _update_statistics(cube)

        full_header = cube.meta.to_fits_header(wcs=cube.wcs)

        hdu_data = fits.CompImageHDU(data=cube.data,
                                     header=full_header,
                                     name="Primary data array")
        hdu_uncertainty = fits.CompImageHDU(data=_pack_uncertainty(cube),
                                            header=full_header,
                                            name="Uncertainty array",
                                            quantize_level=uncertainty_quantize_level)
        hdu_provenance = fits.BinTableHDU.from_columns(fits.ColDefs([fits.Column(
            name="provenance", format="A40", array=np.char.array(cube.meta.provenance))]))
        hdu_provenance.name = "File provenance"

        hdul = cube.wcs.to_fits()
        hdul[0] = fits.PrimaryHDU()
        hdul.insert(1, hdu_data)
        hdul.insert(2, hdu_uncertainty)
        hdul.insert(3, hdu_provenance)
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
    # This is (1/uncertainty_array) * data_array, but this way we save time on memory allocation
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(1, uncertainty_array, out=uncertainty_array)
        np.multiply(data_array, uncertainty_array, out=uncertainty_array)
    return uncertainty_array


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


def load_ndcube_from_fits(path: str | Path, key: str = " ", include_provenance: bool = True) -> NDCube:
    """Load an NDCube from a FITS file."""
    with fits.open(path) as hdul:
        primary_hdu = hdul[1]
        data = primary_hdu.data
        header = primary_hdu.header
        # Reset checksum and datasum to match astropy.io.fits behavior
        header["CHECKSUM"] = ""
        header["DATASUM"] = ""
        meta = NormalizedMetadata.from_fits_header(header)
        if include_provenance:
            meta._provenance = hdul[3].data["provenance"]  # noqa: SLF001
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
