from __future__ import annotations

import os.path
import warnings
from collections import namedtuple, OrderedDict
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
import typing as t

import astropy.units as u
import astropy.wcs.wcsapi
import matplotlib as mpl
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from astropy.io.fits import Header, Card
from dateutil.parser import parse as parse_datetime
from ndcube import NDCube
import yaml

from punchbowl.errors import MissingMetadataError

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_data_path(path):
    return os.path.join(_ROOT, 'data', path)


def load_omniheader(path=None):
    if path is None:
        path = get_data_path("omniheader.csv")
    return pd.read_csv(path, na_filter=False)


def load_level_spec(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_spacecraft_def(path=None):
    if path is None:
        path = get_data_path("spacecraft.yaml")
    with open(path, 'r') as f:
        return yaml.safe_load(f)


HistoryEntry = namedtuple("HistoryEntry", "datetime, source, comment")


class History:
    """Representation of the history of edits done to a PUNCHData object"""

    def __init__(self) -> None:
        self._entries: t.List[HistoryEntry] = []

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


ValueType = t.Union[int, str, float]


class MetaField:
    def __init__(self,
                 keyword: str,
                 comment: str,
                 value: t.Optional[t.Union[int, str, float]],
                 datatype,
                 nullable: bool,
                 mutable: bool,
                 default: t.Optional[t.Union[int, str, float]]):
        if value is not None and not isinstance(value, datatype):
            raise TypeError(f"MetaField value and kind must match. Found kind={datatype} and value={type(value)}.")
        if default is not None and not isinstance(default, datatype):
            raise TypeError(f"MetaField default and kind must match. Found kind={datatype} and default={type(default)}.")
        if len(keyword) > 8:
            raise ValueError("Keywords must be 8 characters or shorter to comply with FITS")
        self._keyword = keyword
        self._comment = comment
        self._value = value
        self._datatype = datatype
        self.nullable = nullable
        self._mutable = mutable
        self._default = default

    @property
    def keyword(self):
        return self._keyword

    @property
    def comment(self):
        return self._comment

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value: ValueType):
        if not self._mutable:
            raise RuntimeError("Cannot mutate this value because it is set to immutable.")
        if isinstance(value, self._datatype):
            self._value = value
        else:
            raise TypeError(f"Value of {self.keyword} was {type(value)} but must be {self._datatype}.")

    @property
    def default(self):
        return self._default

    @default.setter
    def default(self, default: ValueType):
        if isinstance(default, self._datatype):
            self._default = default
        else:
            raise TypeError(f"Value was {type(default)} but must be {self._default}.")


class NormalizedMetadata(Mapping):
    """
    The NormalizedMetadata object standardizes metadata and metadata access in the PUNCH pipeline. It does so by
    making keyword accesses case-insensitive and providing helpful accessors for commonly used formats of the metadata.

    Internally, the keys are always stored as upper-case strings.
    Unlike the FITS standard, keys can be any length string.
    """

    def __len__(self) -> int:
        return sum([len(section) for section in self._contents.values()])


    def __init__(self,
                 contents: t.OrderedDict[str, t.OrderedDict[str, MetaField]],
                 history: t.Optional[History] = None) -> None:

        self._contents = contents
        self._history = history if history is not None else History()

    def __iter__(self):
        return self._contents.__iter__()

    def to_fits_header(self) -> Header:
        hdr = fits.Header()
        for section in self._contents:
            hdr.append(
                ("COMMENT", ("----- " + section + " ").ljust(72, "-")),
                end=True,
            )
            for key, field in self._contents[section].items():
                if field.value is not None:
                    value = field.value
                elif field.value is None and field.nullable:
                    value = field.default
                else:
                    raise RuntimeError(f"Value is null for {field.keyword} and no default is allowed.")
                hdr.append(
                    (
                        field.keyword,
                        value,
                        field.comment,
                    ),
                    end=True,
                )
        return hdr

    @staticmethod
    def _match_product_code_in_level_spec(product_code, level_spec):
        if product_code in level_spec['Products']:
            return level_spec['Products'][product_code]
        else:
            type_code = product_code[:-1]
            found_type_codes = {pc[:-1] for pc in level_spec['Products'].keys()}
            if type_code in found_type_codes:
                return level_spec['Products'][type_code + "?"]
            else:
                raise RuntimeError(f"Product code {product_code} not found in level_spec")

    @staticmethod
    def _load_template_files(omniheader_path, level, level_spec_path, spacecraft, spacecraft_def_path):
        omniheader = load_omniheader(omniheader_path)
        spacecraft_def = load_spacecraft_def(spacecraft_def_path)
        if spacecraft not in spacecraft_def:
            raise RuntimeError(f"Spacecraft {spacecraft} not in spacecraft_def.")

        if level is not None and level_spec_path is not None:
            raise RuntimeError("Only specify the level or level_spec_path, not both.")
        elif level is not None:
            level_spec_path = get_data_path(f"Level{level}.yaml")
            level_spec = load_level_spec(level_spec_path)
        elif level_spec_path is not None:
            level_spec = load_level_spec(level_spec_path)
        else:
            raise RuntimeError("Either level or level_spec_path must be defined. Found None for both.")
        return omniheader, level_spec, spacecraft_def

    @staticmethod
    def _determine_omits_and_overrides(level_spec, product_def):
        this_kinds = product_def['kinds']
        omits, overrides = [], {}
        for section in level_spec['Level']:
            if level_spec['Level'][section] is not None:
                if 'omits' in level_spec['Level'][section]:
                    omits += level_spec['Level'][section]['omits']
                if 'overrides' in level_spec['Level'][section]:
                    for key, value in level_spec['Level'][section]['overrides'].items():
                        overrides[key] = value

        for kind in this_kinds:
            if kind not in level_spec['Kinds']:
                raise RuntimeError(f"{kind} not found in level_spec.")
            if 'omits' in level_spec['Kinds'][kind]:
                omits += level_spec['Kinds'][kind]['omits']
            if 'overrides' in level_spec['Kinds'][kind]:
                for key, value in level_spec['Kinds'][kind]['overrides'].items():
                    overrides[key] = value

        if 'omits' in product_def:
            omits += product_def['omits']

        if 'overrides' in product_def:
            for key, value in product_def['overrides'].items():
                overrides[key] = value

        return omits, overrides

    @classmethod
    def load_template(cls,
                      product_code: str,
                      level: t.Optional[str] = None,
                      level_spec_path: t.Optional[str] = None,
                      omniheader_path: t.Optional[str] = None,
                      spacecraft_def_path: t.Optional[str] = None) -> NormalizedMetadata:
        # load all needed files
        spacecraft = product_code[-1]
        omniheader, level_spec, spacecraft_def = NormalizedMetadata._load_template_files(omniheader_path,
                                                                                         level,
                                                                                         level_spec_path,
                                                                                         spacecraft,
                                                                                         spacecraft_def_path)

        product_def = NormalizedMetadata._match_product_code_in_level_spec(product_code, level_spec)
        omits, overrides = NormalizedMetadata._determine_omits_and_overrides(level_spec, product_def)

        # construct the items to fill
        contents, history = OrderedDict(), History()

        # figure out the sections
        section_rows = np.where(omniheader['TYPE'] == 'section')[0]
        section_titles = omniheader['VALUE'].iloc[section_rows]
        section_ids = omniheader['SECTION'].iloc[section_rows]

        # parse each section
        dtypes = {'str': str, 'int': int, 'float': float}
        for section_id, section_title in zip(section_ids, section_titles):
            if section_title in level_spec['Level']:
                contents[section_title] = OrderedDict()
                for i in np.where(omniheader['SECTION'] == section_id)[0][1:]:
                    e = omniheader.iloc[i]
                    if e['KEYWORD'] not in omits:
                        datatype = dtypes[e['DATATYPE']]
                        value, default = e['VALUE'], e['DEFAULT']
                        if e['KEYWORD'] in overrides:
                            value = overrides[e['KEYWORD']]
                        try:
                            value = datatype(value)
                        except ValueError:
                            raise RuntimeError(f"Value was of the wrong type to parse for {e['KEYWORD']}")
                        finally:
                            if isinstance(value, str):
                                value = value.format(**spacecraft_def[spacecraft])
                        try:
                            default = datatype(default)
                        except ValueError:
                            raise RuntimeError(f"Default was of the wrong type to parse for {e['KEYWORD']}")
                        finally:
                            if isinstance(value, str):
                                default = default.format(**spacecraft_def[spacecraft])
                        contents[section_title][e['KEYWORD']] = MetaField(e['KEYWORD'],
                                                                          e['COMMENT'].format(**spacecraft_def[spacecraft]),
                                                                          value,
                                                                          datatype,
                                                                          e['NULLABLE'],
                                                                          e['MUTABLE'],
                                                                          default)
        return cls(contents, history)

    @property
    def sections(self) -> t.List[str]:
        return list(self._contents.keys())

    @property
    def history(self) -> History:
        return self._history

    @staticmethod
    def _validate_key_is_str(key: str) -> None:
        if not isinstance(key, str):
            raise TypeError(f"Keys for NormalizedMetadata must be strings. You provided {type(key)}.")
        if len(key) > 8:
            raise ValueError("Keys must be <= 8 characters long")

    def __setitem__(self, key: str, value: t.Any) -> None:
        self._validate_key_is_str(key)
        for section_name, section in self._contents.items():
            if key in section:
                self._contents[section_name][key.upper()].value = value
                return

        # reaching here means we haven't returned
        raise RuntimeError(f"MetaField with key={key} not found.")

    def __getitem__(self, key: str) -> t.Any:
        self._validate_key_is_str(key)
        for section_name, section in self._contents.items():
            if key in section:
                return self._contents[section_name][key.upper()]

        # reaching here means we haven't returned
        raise RuntimeError(f"MetaField with key={key} not found.")

    def __delitem__(self, key: str) -> None:
        self._validate_key_is_str(key)
        for section_name, section in self._contents.items():
            if key in section:
                del self._contents[section_name][key.upper()]
                return

        # reaching here means we haven't returned
        raise RuntimeError(f"MetaField with key={key} not found.")

    def __contains__(self, key: str) -> bool:
        self._validate_key_is_str(key)
        for section_name, section in self._contents.items():
            if key in section:
                return True
        return False

    @property
    def product_level(self) -> int:
        if "LEVEL" not in self:
            raise MissingMetadataError("LEVEL is missing from the metadata.")
        return self["LEVEL"].value

    @property
    def datetime(self) -> datetime:
        if "DATE-OBS" not in self:
            raise MissingMetadataError("DATE-OBS is missing from the metadata.")
        return parse_datetime(self["DATE-OBS"].value)


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
        uncertainty: t.Any | None = None,
        mask: t.Any | None = None,
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

        with fits.open(path) as hdul:
            hdu_index = next((i for i, hdu in enumerate(hdul) if hdu.data is not None), 0)
            primary_hdu = hdul[hdu_index]
            header = primary_hdu.header
            data = primary_hdu.data
            meta = NormalizedMetadata(dict(header))  # TODO: make work!
            wcs = WCS(header)
            unit = u.ct

            if len(hdul) > hdu_index + 1:
                secondary_hdu = hdul[hdu_index+1]
                uncertainty = StdDevUncertainty(secondary_hdu.data)
            else:
                uncertainty = None

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
        craft = self.meta["CRAFT"].value
        file_level = self.meta["LEVEL"].value
        type_code = self.meta["TYPECODE"].value
        date_string = self.meta.datetime.strftime("%Y%m%d%H%M%S")
        # TODO: include version number
        return (
            "PUNCH_L" + file_level + "_" + type_code + craft + "_" + date_string
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
        header = self.meta.to_fits_header()

        for entry in self.meta.history:
            header["HISTORY"] = f"{entry.datetime}: {entry.source}, {entry.comment}"

        hdul_list = []

        hdu_dummy = fits.PrimaryHDU()
        hdul_list.append(hdu_dummy)

        hdu_data = fits.CompImageHDU(data=self.data, header=header)
        hdul_list.append(hdu_data)

        if self.uncertainty is not None:
            hdu_uncertainty = fits.CompImageHDU(data = self.uncertainty.array)
            hdul_list.append(hdu_uncertainty)

        hdul = fits.HDUList(hdul_list)

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
