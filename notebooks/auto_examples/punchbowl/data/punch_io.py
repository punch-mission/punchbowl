from __future__ import annotations

import os
import string
import hashlib
import os.path
import warnings
import subprocess
from copy import deepcopy
from typing import Any
from pathlib import Path

import astropy.units as u
import lxml.etree as et
import numpy as np
from astropy.io import fits
from astropy.io.fits import Header
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS, FITSFixedWarning
from glymur import Jp2k, jp2box
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
    file_version = "?" if file_version == "" else file_version  # file version should never be empty!
    return "PUNCH_L" + file_level + "_" + type_code + obscode + "_" + date_string + "_v" + file_version


class DefaultFormatter(string.Formatter):
    """A formatter that doesn't fail if a keyword is missing. Used for quicklook."""

    def get_field(self, field_name: str, args: Any, kwargs: Any) -> str:
        """Provide a special getter that returns the name if it fails."""
        try:
            return super().get_field(field_name, args, kwargs)
        except (KeyError, AttributeError, IndexError):
            return "{" + field_name + "}", ()


def _header_to_xml(header: Header) -> et.Element:
    """
    Convert image header metadata into an XML Tree that can be inserted into a JPEG2000 file header.

    (Helper function adapted from SunPy)
    """
    fits = et.Element("fits")
    already_added = set()
    for key in header:
        if (key in already_added):
            continue
        already_added.add(key)
        el = et.SubElement(fits, key)
        data = header.get(key)
        data = ("1" if data else "0") if isinstance(data, bool) else str(data)
        el.text = data
    return fits


def _generate_jp2_xmlbox(header: Header) -> jp2box.XMLBox:
    """
    Generate the JPEG2000 XML box to be inserted into the JPEG2000 file.

    (Helper function adapted from SunPy)
    """
    header_xml = _header_to_xml(header)
    meta = et.Element("meta")
    meta.append(header_xml)
    tree = et.ElementTree(meta)
    return jp2box.XMLBox(xml=tree)


def write_ndcube_to_quicklook(cube: NDCube,
                              filename: str,
                              layer: int | None = None,
                              vmin: float = 1e-15,
                              vmax: float = 8e-12,
                              include_meta: bool = True,
                              annotation: str | None = None,
                              color: bool = False) -> None:
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
    include_meta : bool
        whether to include metadata in the JPEG2000 file
    annotation : str | None
        a formatted string to add to the bottom of the image as a label
        can access metadata by key, e.g. "typecode={TYPECODE}" would write the data's typecode into the image
    color : bool
        flag to generate RGB quicklook files, grayscale by default

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

    if color:
        mode = "RGBA"
        scaled_arr = (cmap_punch(norm(np.flipud(image))) * 255).astype(np.uint8)
        fill_value = (255, 255, 255)
    else:
        mode = "L"
        scaled_arr = (np.clip(norm(np.flipud(image)), 0, 1) * 255).astype(np.uint8)
        fill_value = 255

    pil_image = Image.fromarray(scaled_arr, mode=mode)

    if annotation:
        pad_height = int(image.shape[1] * 50 / 2048)
        padded_image = Image.new(mode, (pil_image.width, pil_image.height + pad_height))

        padded_image.paste(pil_image, (0, 0))

        draw = ImageDraw.Draw(padded_image)
        font = ImageFont.load_default(size=int(pad_height / 2))

        formatter = DefaultFormatter()
        text = formatter.format(annotation, **cube.meta)
        text_offset = int(10 * image.shape[1] / 2048)
        text_position = (text_offset, pil_image.height + text_offset)
        draw.text(text_position, text, font=font, fill=fill_value)
        pil_image = padded_image

    arr_image = np.array(pil_image)

    tmp_filename = f"{filename}tmp.jp2"
    jp2 = Jp2k(tmp_filename, arr_image)
    meta_boxes = jp2.box
    target_index = len(meta_boxes) - 1
    if include_meta:
        header = cube.meta.to_fits_header(wcs=cube.wcs)
        header.remove("COMMENT", ignore_missing=True, remove_all=True)
        fits_box = _generate_jp2_xmlbox(header)
        meta_boxes.insert(target_index, fits_box)
    jp2.wrap(filename, boxes=meta_boxes)
    os.remove(tmp_filename)


def write_quicklook_to_mp4(files: list[str],
                           filename: str,
                           framerate: int = 5,
                           resolution: int = 1024,
                           codec: str = "libx264",
                           ffmpeg_cmd: str = "ffmpeg",
                           ) -> None:
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
    ffmpeg_cmd : str
        path to the ffmpeg executable

    """
    input_sequence = f"concat:{'|'.join(files)}"

    ffmpeg_command = [
        ffmpeg_cmd,
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
                         skip_wcs_conversion: bool = False,
                         uncertainty_quantize_level: float = 16) -> None:
    """Write an NDCube as a FITS file."""
    if not filename.endswith(".fits"):
        msg = (
            "Filename must have a valid file extension `.fits`"
            f"Found: {os.path.splitext(filename)[1]}"
        )
        raise ValueError(msg)

    if not skip_stats:
        meta = _update_statistics(cube)

    full_header = meta.to_fits_header(wcs=cube.wcs, write_celestial_wcs=not skip_wcs_conversion)
    full_header["FILENAME"] = os.path.basename(filename)

    hdu_data = fits.CompImageHDU(data=cube.data.astype(np.float32) if cube.data.dtype == np.float64 else cube.data,
                                 header=full_header,
                                 name="Primary data array")
    hdu_provenance = fits.BinTableHDU.from_columns(fits.ColDefs([fits.Column(
        name="provenance", format="A40", array=np.char.array(meta.provenance))]))
    hdu_provenance.name = "File provenance"

    hdul = cube.wcs.to_fits()
    hdul[0] = fits.PrimaryHDU()
    hdul.insert(1, hdu_data)
    if meta["LEVEL"].value != "0":
        hdu_uncertainty = fits.CompImageHDU(data=_pack_uncertainty(cube),
                                            header=full_header,
                                            name="Uncertainty array",
                                            quantize_level=uncertainty_quantize_level,
                                            quantize_method=2)
        hdul.insert(2, hdu_uncertainty)
    hdul.append(hdu_provenance)
    hdul.writeto(filename, overwrite=overwrite, checksum=True)
    hdul.close()
    if write_hash:
        write_file_hash(filename)



def _pack_uncertainty(cube: NDCube) -> np.ndarray:
    """Compress the uncertainty for writing to file."""
    return np.zeros_like(cube.data) - 999 if cube.uncertainty is None else 1 / (cube.uncertainty.array / cube.data)


def _unpack_uncertainty(uncertainty_array: np.ndarray, data_array: np.ndarray) -> np.ndarray:
    """Uncompress the uncertainty when reading from a file."""
    # This is (1/uncertainty_array) * data_array, but this way we save time on memory allocation
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(1, uncertainty_array, out=uncertainty_array)
        np.multiply(data_array, uncertainty_array, out=uncertainty_array)
        uncertainty_array[np.isnan(uncertainty_array) * (data_array == 0)] = np.inf
    return uncertainty_array


def _update_statistics(cube: NDCube) -> NormalizedMetadata:
    """Update image statistics in metadata before writing to file."""
    # TODO - Determine DSATVAL omniheader value in calibrated units for L1+

    meta = deepcopy(cube.meta)

    meta["DATAZER"] = len(np.where(cube.data == 0)[0])

    meta["DATASAT"] = len(np.where(cube.data >= meta["DSATVAL"].value)[0])

    nonzero_data = cube.data[np.isfinite(cube.data) * (cube.data != 0)].flatten()

    if len(nonzero_data) > 0:
        meta["DATAAVG"] = np.mean(nonzero_data).item()
        meta["DATAMDN"] = np.median(nonzero_data).item()
        meta["DATASIG"] = np.std(nonzero_data).item()
    else:
        meta["DATAAVG"] = -999.0
        meta["DATAMDN"] = -999.0
        meta["DATASIG"] = -999.0

    percentile_percentages = [1, 10, 25, 50, 75, 90, 95, 98, 99]
    if len(nonzero_data) > 0:
        percentile_values = np.percentile(nonzero_data, percentile_percentages)
        if np.any(np.isnan(percentile_values)):  # report nan if any of the values are nan
            percentile_values = [-999.0 for _ in percentile_percentages]

        for percent, value in zip(percentile_percentages, percentile_values, strict=True):
            meta[f"DATAP{percent:02d}"] = value

        meta["DATAMIN"] = np.min(nonzero_data).item()
        meta["DATAMAX"] = np.max(nonzero_data).item()
    else:
        for percent in percentile_percentages:
            meta[f"DATAP{percent:02d}"] = -999.0

        meta["DATAMIN"] = 0.0
        meta["DATAMAX"] = 0.0

    return meta


def load_ndcube_from_fits(path: str | Path, key: str = " ", include_provenance: bool = True,
                          include_uncertainty: bool = True) -> NDCube:
    """Load an NDCube from a FITS file."""
    with warnings.catch_warnings(), fits.open(path) as hdul:
        warnings.filterwarnings(action="ignore", message=".*CROTA.*Human-readable solar north pole angle.*",
                                category=FITSFixedWarning)
        primary_hdu = hdul[1]
        data = primary_hdu.data
        header = primary_hdu.header
        # Reset checksum and datasum to match astropy.io.fits behavior
        header["CHECKSUM"] = ""
        header["DATASUM"] = ""
        meta = NormalizedMetadata.from_fits_header(header)
        if include_provenance:
            if isinstance(hdul[-1], fits.hdu.BinTableHDU):
                meta._provenance = hdul[-1].data["provenance"]  # noqa: SLF001
            else:
                msg = "Provenance HDU does not appear to be BinTableHDU type."
                raise ValueError(msg)
        wcs = WCS(header, hdul, key=key)
        unit = u.ct

        if include_uncertainty and len(hdul) >= 3 and isinstance(hdul[2], fits.hdu.CompImageHDU):
            secondary_hdu = hdul[2]
            uncertainty = _unpack_uncertainty(secondary_hdu.data.astype(float), data)
            uncertainty = StdDevUncertainty(uncertainty)
        else:
            uncertainty = None

    return NDCube(
        data.view(dtype=data.dtype.newbyteorder()).byteswap().astype(float),
        wcs=wcs,
        uncertainty=uncertainty,
        meta=meta,
        unit=unit,
    )
