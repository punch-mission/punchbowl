from __future__ import annotations

import os.path
import warnings
from collections import namedtuple
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import astropy.units as u
import astropy.wcs.wcsapi
import matplotlib as mpl
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from dateutil.parser import parse as parse_datetime
from ndcube import NDCube

from punchbowl.errors import MissingMetadataError

HistoryEntry = namedtuple("HistoryEntry", "datetime, source, comment")


class History:
    """Representation of the history of edits done to a PUNCHData object"""

    def __init__(self) -> None:
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

    def add_now(self, source: str, comment: str) -> None:
        """Adds a new history entry at the current time.

        Parameters
        ----------
        source : str
            what module of the code the history entry originates from
        comment : str
            a note of what the history comment means

        Returns
        -------
        None
        """
        self._entries.append(HistoryEntry(datetime.now(), source, comment))

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

    def __iter__(self) -> History:
        self.current_index = 0
        return self

    def __next__(self) -> HistoryEntry:
        if self.current_index >= len(self):
            raise StopIteration
        entry = self._entries[self.current_index]
        self.current_index += 1
        return entry  # noqa:  RET504


class NormalizedMetadata(Mapping):
    """
    The NormalizedMetadata object standardizes metadata and metadata access in the PUNCH pipeline. It does so by
    making keyword accesses case-insensitive and providing helpful accessors for commonly used formats of the metadata.

    Internally, the keys are always stored as upper-case strings.
    Unlike the FITS standard, keys can be any length string.
    """
    def __init__(self, contents: dict[str, Any], required_fields: Optional[Set[str]] = None) -> None:
        # Validate all the keys are acceptable
        for key in contents:
            self._validate_key_is_str(key)
            if key.upper() == "HISTORY-OBJECT":
                raise KeyError("HISTORY-OBJECT is a reserved keyword for NormalizedMetadata. "
                               "It cannot be in a passed in contents.")

        # Create the contents now
        self._contents = {k.upper(): v for k, v in contents.items()}

        self.required_fields = required_fields if required_fields is not None else set()
        self.validate_required_fields()

        # Add a history object
        self._contents["HISTORY-OBJECT"] = History()

    def validate_required_fields(self):
        # Check that all required fields are present
        if self.required_fields is not None:
            for key in self.required_fields:
                if key.upper() not in self._contents:
                    raise RuntimeError(f"{key} is missing."
                                       f"According to required_keys it must be a key in NormalizedMetadata.")

    def __iter__(self) -> Dict:
        return self._contents.__iter__()

    @property
    def history(self) -> History:
        return self._contents["HISTORY-OBJECT"]

    @staticmethod
    def _validate_key_is_str(key: str) -> None:
        if not isinstance(key, str):
            raise TypeError(f"Keys for NormalizedMetadata must be strings. You provided {type(key)}.")

    def __setitem__(self, key: str, value: Any) -> None:
        self._validate_key_is_str(key)
        self._contents[key.upper()] = value

    def __getitem__(self, key: str) -> Any:
        self._validate_key_is_str(key)
        return self._contents[key.upper()]

    def __delitem__(self, key: str) -> None:
        self._validate_key_is_str(key)
        if key.upper() in self.required_fields:
            raise ValueError(f"Cannot delete a required_field: {key}")
        else:
            del self._contents[key.upper()]

    def __contains__(self, key: str) -> bool:
        self._validate_key_is_str(key)
        return key.upper() in self._contents

    def __len__(self) -> int:
        return len(self._contents) - 1  # we subtract 1 to ignore the history object

    @property
    def product_level(self) -> int:
        if "LEVEL" not in self._contents:
            raise MissingMetadataError("LEVEL is missing from the metadata.")
        return self._contents["LEVEL"]

    @property
    def datetime(self) -> datetime:
        if "DATE-OBS" not in self._contents:
            raise MissingMetadataError("DATE-OBS is missing from the metadata.")
        return parse_datetime(self._contents["DATE-OBS"])


HEADER_TEMPLATE_COLUMNS = ["TYPE", "KEYWORD", "VALUE", "COMMENT", "DATATYPE", "STATE"]


class HeaderTemplate:
    """PUNCH data object header template
    Class to generate a PUNCH data object header template, along with associated methods.

    - TODO : make custom types of warnings more specific so that they can be filtered
    """

    def __init__(self, template: pd.DataFrame=None) -> None:
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

        Parameters
        ----------
        meta : NormalizedMetadata
            the contents of the metadata you wish to fill with

        Returns
        -------
        astropy.io.fits.header
            Header with filled fields
        """
        hdr = fits.Header()

        type_converter = {"str": str, "int": int, "float": float}

        for _, entry in self._table.iterrows():
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
            if key in hdr and key in empty_keywords and value != "":  # only update if it's not the empty string
                hdr[key] = value
                empty_keywords.remove(key)

        if empty_keywords:
            warnings.warn(f"Some keywords left empty: {empty_keywords}", RuntimeWarning)

        return hdr

    def find_empty(self) -> list[str]:
        """Return a list of empty required header keywords.

        Returns
        -------
        list[str]
            List of unassigned keywords
        """
        empty_keywords = []
        for _, row in self._table.iterrows():
            if row["TYPE"] == "keyword" and not row["VALUE"]:
                empty_keywords.append(row["KEYWORD"])
        return empty_keywords


PUNCH_REQUIRED_META_FIELDS = {"OBSRVTRY", "LEVEL", "TYPECODE", "DATE-OBS"}


class PUNCHData(NDCube):
    """PUNCH data object

    PUNCHData is essentially a normal ndcube with a StandardizedMetadata and some helpful methods.

    See Also
    --------
    NDCube : Base container for the PUNCHData object
    """

    def __init__(
        self,
        data: np.ndarray,
        wcs: astropy.wcs.wcsapi.BaseLowLevelWCS
        | astropy.wcs.wcsapi.BaseHighLevelWCS,
        meta: NormalizedMetadata,
        uncertainty: Any | None = None,
        mask: Any | None = None,
        unit: astropy.units.Unit = None,
        copy: bool = False,
        **kwargs,
    ) -> None:
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
        kwargs
            Additional keyword arguments

        Notes
        -----
        As the PUNCHData object is a subclass of NDCube, the constructor follows much of the same form.

        PUNCHData objects also contain history information and have special functionality for manipulating PUNCH data.
        """
        if not PUNCH_REQUIRED_META_FIELDS.issubset(meta.required_fields):
            missing = {field for field in PUNCH_REQUIRED_META_FIELDS if field not in meta.required_fields}
            raise ValueError(f"Required fields of meta should contain all PUNCH required fields. Missing {missing}.")

        super().__init__(
            data,
            wcs=wcs,
            uncertainty=uncertainty,
            mask=mask,
            meta=meta,
            unit=unit,
            copy=copy,
            **kwargs,
        )

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

        # TODO: This could be done more elegantly in a future task
        # TODO: Below, are the units always counts for the primary data array????

        with fits.open(path) as hdul:
            # A one-length hdul can be a primary uncompressed data array
            if len(hdul) == 1:
                data = hdul[0].data
                meta = NormalizedMetadata(dict(hdul[0].header), required_fields=PUNCH_REQUIRED_META_FIELDS)
                wcs = WCS(hdul[0].header)
                uncertainty = None
                unit = u.ct  # counts
            # A two-length hdul can be a compressed primary data array, or uncompressed data and uncertainty arrays
            elif len(hdul) == 2:
                # Compressed primary data array
                if isinstance(hdul[1], fits.CompImageHDU):
                    data = hdul[1].data
                    meta = NormalizedMetadata(dict(hdul[1].header), required_fields=PUNCH_REQUIRED_META_FIELDS)
                    wcs = WCS(hdul[1].header)
                    uncertainty = None
                    unit = u.ct  # counts
                # Uncompressed data and uncertainty arrays
                else:
                    data = hdul[0].data
                    meta = NormalizedMetadata(dict(hdul[0].header), required_fields=PUNCH_REQUIRED_META_FIELDS)
                    wcs = WCS(hdul[0].header)
                    uncertainty = StdDevUncertainty(hdul[1].data)
                    unit = u.ct  # counts
            # A three-length hdul can be a primary compressed data and uncertainty arrays
            elif len(hdul) == 3:
                data = hdul[1].data
                meta = NormalizedMetadata(dict(hdul[1].header), required_fields=PUNCH_REQUIRED_META_FIELDS)
                wcs = WCS(hdul[1].header)
                uncertainty = StdDevUncertainty(hdul[2].data)
                unit = u.ct  # counts

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
    def filename_base(self) -> str:
        """Dynamically generate an id string for the given data product, using the format 'Ln_ttO_yyyymmddhhmmss'

        Returns
        -------
        str
            output identification string
        """
        observatory = str(self.meta["OBSRVTRY"])
        file_level = str(self.meta["LEVEL"])
        type_code = str(self.meta["TYPECODE"])
        date_string = str(self.meta.datetime.strftime("%Y%m%d%H%M%S"))
        # TODO: include version number
        return "PUNCH_L" + file_level + "_" + type_code + observatory + "_" + date_string

    @property
    def is_blank(self) -> bool:
        return self.meta['BLANK'] if 'BLANK' in self.meta else False

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
        elif filename.endswith((".png", ".jpg", ".jpeg")):
            self._write_ql(filename, overwrite=overwrite)
        else:
            raise ValueError(
                "Filename must have a valid file extension (.fits, .png, .jpg, .jpeg). "
                f"Found: {os.path.splitext(filename)[1]}"
            )

    def _write_fits(self, filename: str, overwrite: bool=True) -> None:
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

        for entry in self.meta.history:
            header["HISTORY"] = f"{entry.datetime}: {entry.source}, {entry.comment}"

        hdu_data = fits.PrimaryHDU(data=self.data, header=header)

        # TODO : Make an uncertainty header
        hdu_uncertainty = fits.ImageHDU()
        hdu_uncertainty.data = self.uncertainty.array

        hdul = fits.HDUList([hdu_data, hdu_uncertainty])

        # Write to FITS
        hdul.writeto(filename, overwrite=overwrite)


    def _write_ql(self, filename: str, overwrite: bool = True) -> None:
        """Write an 8-bit scaled version of the specified data array to a PNG file

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
        if os.path.isfile(filename) and not overwrite:
            raise OSError(f"File {filename} already exists."
                           "If you mean to replace it then use the argument 'overwrite=True'.")


        if self.data.ndim != 2:
            raise ValueError("Specified output data should have two-dimensions.")

        # Scale data array to 8-bit values
        output_data = int(
            np.fix(
                np.interp(
                    self.data, (self.data.min(), self.data.max()), (0, 2**8 - 1)
                )
            )
        )

        # Write image to file
        mpl.image.saveim(filename, output_data)

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
            template_path = str(Path(__file__).parent /
                                f"data/HeaderTemplate/HeaderTemplate_L{self.meta.product_level}.csv")

        template = HeaderTemplate.load(template_path)
        return template.fill(self.meta)


    def duplicate_with_updates(self, data: np.ndarray=None,
                               wcs: astropy.wcs.WCS= None,
                               uncertainty: np.ndarray=None,
                               meta=None,
                               unit=None) -> PUNCHData:
        """Copies a PUNCHData. Any field specified in the call is modified. All others are a direct copy. """
        return PUNCHData(data=data if data is not None else self.data,
                         wcs=wcs if wcs is not None else self.wcs,
                         uncertainty=uncertainty if uncertainty is not None else self.uncertainty,
                         meta=meta if meta is not None else self.meta,
                         unit=unit if unit is not None else self.unit
                         )
