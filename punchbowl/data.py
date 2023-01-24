from __future__ import annotations

import warnings
import os.path
from collections import namedtuple
from collections.abc import Mapping
from datetime import datetime
from typing import Union, List, Dict, Any, Iterator
from dateutil.parser import parse as parse_datetime
from pathlib import Path


import matplotlib
import numpy as np
import astropy.wcs.wcsapi
import astropy.units as u
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube
import pandas as pd

from punchbowl.errors import MissingMetadataError

HistoryEntry = namedtuple("HistoryEntry", "datetime, source, comment")


class History:
    """Representation of the history of edits done to a PUNCHData object"""

    def __init__(self):
        self._entries: List[HistoryEntry] = []

    def add_entry(self, entry: HistoryEntry) -> None:
        """
        Add an entry to the History log

        Parameters
        ----------
        entry : HistoryEntry
            A HistoryEntry object to add to the History log

        Returns
        -------
        None

        """
        self._entries.append(entry)

    def clear(self) -> None:
        """
        Clears all the history entries so the History is blank

        Returns
        -------
        None
        """
        self._entries = []

    def __getitem__(self, index: int) -> HistoryEntry:
        """
        Given an index, returns the requested HistoryEntry

        Parameters
        ----------
        index : int
            numerical index of the history entry, increasing number typically indicates an older entry

        Returns
        -------
        HistoryEntry
            history at specified `index`
        """
        return self._entries[index]

    def most_recent(self) -> HistoryEntry:
        """
        Gets the most recent HistoryEntry, i.e. the youngest

        Returns
        -------
        HistoryEntry
            HistoryEntry that is the youngest
        """
        return self._entries[-1]

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            the number of history entries
        """
        return len(self._entries)

    def __str__(self) -> str:
        """
        Formats a string combining all the history entries

        Returns
        -------
        str
            a combined record of the history entries
        """
        return "\n".join(
            [f"{e.datetime}: {e.source}: {e.comment}" for e in self._entries]
        )

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= len(self):
            raise StopIteration
        entry = self._entries[self.current_index]
        self.current_index += 1
        return entry


class NormalizedMetadata(Mapping):
    """
    The NormalizedMetadata object standardizes metadata and metadata access in the PUNCH pipeline. It does so by
    making keyword accesses case-insensitive and providing helpful accessors for commonly used formats of the metadata.

    Internally, the keys are always stored as upper-case strings.
    Unlike the FITS standard, keys can be any length string.
    """
    def __init__(self, contents: dict[str, Any]):
        for key in contents:
            self._validate_key_is_str(key)
        self._contents = {k.upper(): v for k, v in contents.items()}

    def __iter__(self):
        return self._contents.__iter__()

    @staticmethod
    def _validate_key_is_str(key):
        if not isinstance(key, str):
            raise TypeError(f"Keys for NormalizedMetadata must be strings. You provided {type(key)}.")
    def __setitem__(self, key: str, value: Any):
        self._validate_key_is_str(key)
        self._contents[key.upper()] = value

    def __getitem__(self, key: str) -> Any:
        self._validate_key_is_str(key)
        return self._contents[key.upper()]

    def __delitem__(self, key: str) -> None:
        self._validate_key_is_str(key)
        del self._contents[key.upper()]

    def __contains__(self, key: str) -> bool:
        self._validate_key_is_str(key)
        return key.upper() in self._contents

    def __len__(self) -> int:
        return len(self._contents)


HEADER_TEMPLATE_COLUMNS = ["TYPE", "KEYWORD", "VALUE", "COMMENT", "DATATYPE", "STATE"]

class HeaderTemplate:
    """PUNCH data object header template
    Class to generate a PUNCH data object header template, along with associated methods.

    - TODO : make custom types of warnings more specific so that they can be filtered
    """

    def __init__(self, template=None):
        self._table = (
            pd.DataFrame(columns=HEADER_TEMPLATE_COLUMNS)
            if template is None
            else template
        )
        if not np.all(self._table.columns.values == HEADER_TEMPLATE_COLUMNS):
            raise ValueError(
                f"HeaderTemplate must have columns {HEADER_TEMPLATE_COLUMNS}"
                f"Found: {self._table.columns.values}"
            )

    @classmethod
    def load(cls, path: str) -> HeaderTemplate:
        """Loads an input template file to generate a header object.

        Parameters
        ----------
        path
            path to input header template file

        Returns
        -------
        HeaderTemplate
            header template with data from a specified CSV
        """

        if path.endswith(".csv"):
            template = HeaderTemplate(template=pd.read_csv(path, keep_default_na=False))
        else:
            raise ValueError(
                "Header template must be a CSV file."
                f"Found {os.path.splitext(path)[1]} file"
            )

        return template

    def fill(self, meta: NormalizedMetadata) -> fits.Header:
        """Parses an input template header comma separated value (CSV) file to generate an astropy header object.
        # TODO: update

        Parameters
        ----------
        path
            input filename

        Returns
        -------
        astropy.io.fits.header
            Header with filled fields
        """
        hdr = fits.Header()

        type_converter = {"str": str, "int": int, "float": np.float}

        for row_i, entry in self._table.iterrows():
            if entry["TYPE"] == "section":
                if len(entry["COMMENT"]) > 72:
                    warnings.warn(
                        "Section text exceeds 80 characters, EXTEND will be used.",
                        RuntimeWarning,
                    )
                hdr.append(
                    ("COMMENT", ("----- " + entry["COMMENT"] + " ").ljust(72, "-")),
                    end=True,
                )

            elif entry["TYPE"] == "comment":
                hdr.append(("COMMENT", entry["VALUE"]), end=True)

            elif entry["TYPE"] == "keyword":
                if len(entry["VALUE"]) + len(entry["COMMENT"]) > 72:
                    warnings.warn(
                        "Section text exceeds 80 characters, EXTEND will be used.",
                        RuntimeWarning,
                    )

                hdr.append(
                    (
                        entry["KEYWORD"],
                        type_converter[entry["DATATYPE"]](entry["VALUE"])
                        if entry["VALUE"]
                        else "",
                        entry["COMMENT"],
                    ),
                    end=True,
                )

        empty_keywords = set(self.find_empty())
        for key, value in meta.items():
            if key in hdr and key in empty_keywords:
                if value != "":  # only update if it's not the empty string
                    hdr[key] = value
                    empty_keywords.remove(key)

        if empty_keywords:
            warnings.warn(f"Some keywords left empty: {empty_keywords}", RuntimeWarning)

        return hdr

    def find_empty(self) -> list:
        """Return a list of empty required header keywords.

        Returns
        -------
        List
            List of unassigned keywords
        """
        empty_keywords = []
        for row_i, row in self._table.iterrows():
            if row["TYPE"] == "keyword":
                if not row["VALUE"]:
                    empty_keywords.append(row["KEYWORD"])
        return empty_keywords


class PUNCHData(NDCube):
    """PUNCH data object

    See Also
    --------
    NDCube : Base container for the PUNCHData object
    """

    def __init__(
        self,
        data: np.ndarray,
        wcs: astropy.wcs.wcsapi.BaseLowLevelWCS
        | astropy.wcs.wcsapi.BaseHighLevelWCS
        | None = None,
        uncertainty: Any | None = None,
        mask: Any | None = None,
        meta: NormalizedMetadata | None = None,
        unit: astropy.units.Unit = None,
        copy: bool = False,
        history: History | None = None,
        **kwargs,
    ):
        """Initialize PUNCH Data

        Parameters
        ----------
        data
            Primary observation data (2D or multidimensional ndarray)
        wcs
            World coordinate system object describing observation data axes
        uncertainty
            Measure of pixel uncertainty mapping from the primary data array
        mask
            Boolean mapping of invalid pixels mapping from the primary data array (True = masked out invalid pixels)
        meta
            Observation metadata, comprised of keywords and values as an astropy FITS header object
        unit
            Units for the measurements in the primary data array
        copy
            Create arguments as a copy (True), or as a reference where possible (False, default)
        history
            Reference of the history of a given data object (modules passed through, etc)
        kwargs
            Additional keyword arguments

        Notes
        -----
        As the PUNCHData object is a subclass of NDCube, the constructor follows much of the same form.

        PUNCHData objects also contain history information and have special functionality for manipulating PUNCH data.
        """
        super().__init__(
            data,
            wcs=wcs,
            uncertainty=uncertainty,
            mask=mask,
            meta=meta if meta is not None else NormalizedMetadata(dict()),
            unit=unit,
            copy=copy,
            **kwargs,
        )
        self._history = history if history else History()

    def add_history(self, time: datetime, source: str, comment: str) -> None:
        """Log a new history entry

        Parameters
        ----------
        time
            time the history update occurred
        source
            module that the history update originates from
        comment
            explanation of what happened

        Returns
        -------
        None
        """
        self._history.add_entry(HistoryEntry(time, source, comment))

    @classmethod
    def from_fits(cls, path: str) -> PUNCHData:
        """Populates a PUNCHData object from specified FITS file.

        Parameters
        ----------
        path
            filename from which to generate a PUNCHData object

        Returns
        -------
        PUNCHData
            loaded object
        """

        with fits.open(path) as hdul:
            data = hdul[0].data
            meta = NormalizedMetadata(dict(hdul[0].header))
            wcs = WCS(hdul[0].header)
            uncertainty = StdDevUncertainty(hdul[1].data)
            unit = u.ct  # counts
            # TODO: is the unit always counts????

        return cls(
            data.newbyteorder().byteswap(inplace=True),
            wcs=wcs,
            uncertainty=uncertainty,
            meta=meta,
            unit=unit,
        )

    @property
    def weight(self) -> np.ndarray:
        """Generate a corresponding weight map from the uncertainty array

        Returns
        -------
        np.ndarray
            weight map computed from uncertainty array
        """

        return 1.0 / self.uncertainty.array

    @property
    def id(self) -> str:
        """Dynamically generate an id string for the given data product, using the format 'Ln_ttO_yyyymmddhhmmss'

        Returns
        -------
        str
            output identification string
        """
        observatory = self.meta["OBSRVTRY"]
        file_level = self.meta["LEVEL"]
        type_code = self.meta["TYPECODE"]
        date_string = self.datetime.strftime("%Y%m%d%H%M%S")
        return (
            "PUNCH_L" + file_level + "_" + type_code + observatory + "_" + date_string
        )

    def write(self, filename: str, overwrite=True) -> None:
        """Write PUNCHData elements to file

        Parameters
        ----------
        filename
            output filename (including path and file extension), extension must be .fits, .png, .jpg, or .jpeg
        overwrite
            True will overwrite an exiting file, False will create an exception if a file exists

        Returns
        -------
        None

        Raises
        -----
        ValueError
            If `filename` does not end in .fits, .png, .jpg, or .jpeg

        """

        if filename.endswith(".fits"):
            self._write_fits(filename, overwrite=overwrite)
        elif filename.endswith(".png"):
            self._write_ql(filename)
        elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
            self._write_ql(filename)
        else:
            raise ValueError(
                "Filename must have a valid file extension (.fits, .png, .jpg, .jpeg). "
                f"Found: {os.path.splitext(filename)[1]}"
            )

    def _write_fits(self, filename: str, overwrite=True) -> None:
        """Write PUNCHData elements to FITS files

        Parameters
        ----------
        filename
            output filename (including path and file extension)
        overwrite
            True will overwrite an exiting file, False will throw an exception in that scenario

        Returns
        -------
        None
        """
        header = self.create_header(None)  # uses template_path=none so the pipeline selects one based on metadata

        for entry in self._history:
            header["HISTORY"] = f"{entry.datetime}: {entry.source}, {entry.comment}"

        hdu_data = fits.PrimaryHDU(data=self.data, header=header)

        # TODO : Make an uncertainty header
        hdu_uncertainty = fits.ImageHDU()
        hdu_uncertainty.data = self.uncertainty.array

        hdul = fits.HDUList([hdu_data, hdu_uncertainty])

        # Write to FITS
        hdul.writeto(filename, overwrite=overwrite)


    def _write_ql(self, filename: str) -> None:
        """Write an 8-bit scaled version of the specified data array to a PNG file

        Parameters
        ----------
        filename
            output filename (including path and file extension)

        Returns
        -------
        None
        """

        if self.data.ndim != 2:
            raise ValueError("Specified output data should have two-dimensions.")

        # Scale data array to 8-bit values
        output_data = np.int(
            np.fix(
                np.interp(
                    self.data, (self.data.min(), self.data.max()), (0, 2**8 - 1)
                )
            )
        )

        # Write image to file
        matplotlib.image.saveim(filename, output_data)

    def create_header(self, template_path: str = None) -> fits.Header:
        """
        Validates / generates PUNCHData object metadata using data product header standards

        Parameters
        ----------
        template_path
            specified header template file with which to validate

        Returns
        -------
        fits.Header
            a full constructed and filled FITS header that reflects the data
        """
        if template_path is None:
            template_path = str(Path(__file__).parent / f"data/HeaderTemplate/HeaderTemplate_L{self.product_level}.csv")

        template = HeaderTemplate.load(template_path)
        return template.fill(self.meta)

    @property
    def product_level(self) -> int:
        if 'LEVEL' in self.meta:
            return self.meta['LEVEL']
        else:
            raise MissingMetadataError("LEVEL is missing from the metadata.")

    @property
    def datetime(self) -> datetime:
        return parse_datetime(self.meta["date-obs"])

    def duplicate_with_updates(self, data: np.ndarray=None,
                               wcs: astropy.wcs.WCS= None,
                               uncertainty: np.ndarray=None,
                               meta=None,
                               history=None,
                               unit=None) -> PUNCHData:
        return PUNCHData(data=data if data is not None else self.data,
                         wcs=wcs if wcs is not None else self.wcs,
                         uncertainty=uncertainty if uncertainty is not None else self.uncertainty,
                         meta=meta if meta is not None else self.meta,
                         history=history if history is not None else self._history,
                         unit=unit if unit is not None else self.unit
                         )
