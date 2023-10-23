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
            raise TypeError(f"Value was {type(value)} but must be {self._datatype}.")

    @property
    def default(self):
        return self._default

    @default.setter
    def default(self, default: ValueType):
        if isinstance(default, self._datatype):
            self._default = default
        else:
            raise TypeError(f"Value was {type(default)} but must be {self._default}.")


class NormalizedMetadata:
    """
    The NormalizedMetadata object standardizes metadata and metadata access in the PUNCH pipeline. It does so by
    making keyword accesses case-insensitive and providing helpful accessors for commonly used formats of the metadata.

    Internally, the keys are always stored as upper-case strings.
    Unlike the FITS standard, keys can be any length string.
    """
    def __init__(self,
                 contents: t.OrderedDict[str, t.OrderedDict[str, MetaField]],
                 history: t.Optional[History] = None) -> None:

        self._contents = contents
        self._history = history if history is not None else History()

        # # Validate all the keys are acceptable
        # for key in contents:
        #     self._validate_key_is_str(key)
        #     if key.upper() == "HISTORY-OBJECT":
        #         raise KeyError("HISTORY-OBJECT is a reserved keyword for NormalizedMetadata. "
        #                        "It cannot be in a passed in contents.")
        #
        # # Create the contents now
        # self._contents = {k.upper(): v for k, v in contents.items()}
        #
        # self.required_fields = required_fields if required_fields is not None else set()
        # self.validate_required_fields()
        #
        # # Add a history object
        # self._contents["HISTORY-OBJECT"] = History()

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

        if level is not None and omniheader is not None:
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
                            raise RuntimeError("Value was of the wrong type to parse")
                        finally:
                            if isinstance(value, str):
                                value = value.format(**spacecraft_def[spacecraft])
                        try:
                            default = datatype(default)
                        except ValueError:
                            raise RuntimeError("Default was of the wrong type to parse")
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
                self._contents[section_name][key.upper()] = value
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


    # @property
    # def product_level(self) -> int:
    #     if "LEVEL" not in self._contents:
    #         raise MissingMetadataError("LEVEL is missing from the metadata.")
    #     return self._contents["LEVEL"]
    #
    # @property
    # def datetime(self) -> datetime:
    #     if "DATE-OBS" not in self._contents:
    #         raise MissingMetadataError("DATE-OBS is missing from the metadata.")
    #     return parse_datetime(self._contents["DATE-OBS"])
#
#
# HEADER_TEMPLATE_COLUMNS = ["TYPE", "KEYWORD", "VALUE", "COMMENT", "DATATYPE", "STATE"]
#
#
# class HeaderTemplate:
#     """PUNCH data object header template
#     Class to generate a PUNCH data object header template, along with associated methods.
#
#     - TODO : make custom types of warnings more specific so that they can be filtered
#     """
#
#     def __init__(self, omniheader: pd.DataFrame, level_definition: Dict, product_definition: Dict) -> None:
#         self.omniheader = omniheader
#         self.level_definition = level_definition
#         self.product_definition = product_definition
#
#     @classmethod
#     def from_file(cls, omniheader_path: str, definition_path: str, product_code: str):
#         omniheader = pd.read_csv(omniheader_path, na_filter=False)
#         with open(definition_path, 'r') as f:
#             contents = yaml.safe_load(f)
#         return cls(omniheader, contents['LEVEL'], contents[product_code])
#
#
#     @staticmethod
#     def merge_keys(omniheader, level_definition, product_definition):
#         """Merge two sets of FITS header recipes
#
#         Parameters
#         ----------
#         omniheader : DataFrame
#             Omnibus file of all header values, everywhere
#         level_definition : Dict
#             First dictionary of keywords / sections
#         product_definition : Dict
#             Second dictionary of keywords, to be merged into the first
#
#         Returns
#         -------
#         Dict
#             Merged section and keyword dictionary
#         """
#         output_keys = level_definition.copy()
#
#         for key in product_definition:
#             section_num = omniheader.loc[omniheader['KEYWORD'] == key]['SECTION'].values[0]
#             section_str = (omniheader.loc[omniheader['SECTION'] == section_num]['VALUE'].values[0]).strip('- ')
#
#             if output_keys[section_str]:
#                 output_keys[section_str] = {key: product_definition[key]}
#             else:
#                 output_keys[section_str] = output_keys[section_str] | {key: product_definition[key]}
#
#         return output_keys
#
#     @staticmethod
#     def _prepare_row_output(row, metadata: NormalizedMetadata) -> Union[int, float, np.double]:
#         conversion_func = {'int': int, 'float': float, 'double': np.double, 'str': str}
#         if row['VALUE'] != '':
#             return conversion_func[row['DATATYPE']](row['VALUE'])
#         else:
#             if row['KEYWORD'] not in metadata:
#                 raise MissingMetadataError(f"{row['KEYWORD']} was not found in the metadata.")
#             else:
#                 return metadata[row['KEYWORD']]
#
#     def fill(self, metadata: NormalizedMetadata):
#         output_header = Header()
#
#         output_keys = HeaderTemplate.merge_keys(self.omniheader, self.level_definition, self.product_definition)
#
#         for i in np.unique(self.omniheader['SECTION']):
#             section_df = self.omniheader.loc[self.omniheader['SECTION'] == i]
#             section_str = section_df.iloc[0]['VALUE'].strip('- ')
#
#             if section_str in output_keys:
#                 for _, row in section_df.iterrows():
#                     if row['TYPE'] == 'section':
#                         output_header.append(('COMMENT', row['VALUE']))
#                     elif row['TYPE'] == 'keyword':
#                         value = HeaderTemplate._prepare_row_output(row, metadata)
#
#                         # if we want to include the full section
#                         if isinstance(output_keys[section_str], bool) and output_keys[section_str]:
#                             output_header.append((row['KEYWORD'], value, row['COMMENT']), end=True)
#                         # else if the section has only some keywords included, potentially with modifications
#                         elif isinstance(output_keys[section_str], dict):
#                             # if the keyword has an overwritten value
#                             if (row['KEYWORD'] in output_keys[section_str]
#                                and not isinstance(output_keys[section_str][row['KEYWORD']], bool)):
#                                 output_header.append((row['KEYWORD'],
#                                                       output_keys[section_str][row['KEYWORD']],
#                                                       row['COMMENT']), end=True)
#                             else:  # the keyword is dynamic or not overwritten
#                                 output_header.append((row['KEYWORD'], value, row['COMMENT']), end=True)
#                         else:
#                             raise RuntimeError("The header template is poorly formed.")
#         return output_header
#

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
        observatory = self.meta["OBSRVTRY"]
        file_level = self.meta["LEVEL"]
        type_code = self.meta["TYPECODE"]
        date_string = self.meta.datetime.strftime("%Y%m%d%H%M%S")
        # TODO: include version number
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

    # def create_header(self,  omnibus_path: str = None, template_path: str = None) -> fits.Header:
    #     """
    #     Validates / generates PUNCHData object metadata using data product header standards
    #
    #     Parameters
    #     ----------
    #     template_path
    #         specified header template file with which to validate
    #
    #     Returns
    #     -------
    #     fits.Header
    #         a full constructed and filled FITS header that reflects the data
    #     """
    #     if template_path is None:
    #         template_path = str(Path(__file__).parent /
    #                             f"data/HeaderTemplate/Level{self.meta.product_level}.yaml")
    #
    #     if omnibus_path is None:
    #         omnibus_path = str(Path(__file__).parent / "data/HeaderTemplate/omniheader.csv")
    #
    #     template = HeaderTemplate.from_file(omnibus_path,
    #                                         template_path,
    #                                         f"{self.meta['TYPECODE']}{self.meta['OBSRVTRY']}")
    #     return template.fill(self.meta)

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
