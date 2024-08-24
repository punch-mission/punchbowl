from __future__ import annotations

import os
import typing as t
import warnings
from datetime import datetime
from collections import OrderedDict
from collections.abc import Mapping

import astropy.units as u
import numpy as np
import pandas as pd
import yaml
from astropy.coordinates import GCRS, SkyCoord
from astropy.io import fits
from astropy.io.fits import Header
from astropy.time import Time
from astropy.wcs import WCS
from dateutil.parser import parse as parse_datetime
from sunpy.coordinates import frames, sun
from sunpy.coordinates.sun import _sun_north_angle_to_z
from sunpy.map import solar_angular_radius

from punchbowl.data.history import History
from punchbowl.data.wcs import calculate_celestial_wcs_from_helio
from punchbowl.exceptions import ExtraMetadataWarning, MissingMetadataError

ValueType = int | str | float
_ROOT = os.path.abspath(os.path.dirname(__file__))
REQUIRED_HEADER_KEYWORDS = ["SIMPLE", "BITPIX", "NAXIS", "EXTEND"]
DISTORTION_KEYWORDS = ["CPDIS1", "CPDIS2", "DP1", "DP2"]


def load_omniheader(path: str | None = None) -> pd.DataFrame:
    """Load full metadata specifications."""
    if path is None:
        path = os.path.join(_ROOT, "data", "omniheader.csv")
    return pd.read_csv(path, na_filter=False)


def load_level_spec(path: str) -> dict[str, t.Any]:
    """Load data product metadata specifications."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_spacecraft_def(path: str | None = None) -> dict[str, t.Any]:
    """
    Load spacecraft metadata specifications.

    If path is None, then it loads a default from the package.
    """
    if path is None:
        path = os.path.join(_ROOT, "data", "spacecraft.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


class MetaField:
    """The MetaField object describes a single field within the NormalizedMetadata object."""

    def __init__(
        self,
        keyword: str,
        comment: str,
        value: ValueType | None,
        datatype: t.Any,
        nullable: bool,
        mutable: bool,
        default: ValueType | None,
    ) -> None:
        """
        Create a MetaField.

        Parameters
        ----------
        keyword: str
            FITS keyword for this field
        comment: str
            FITS compliant comment for this field
        value : int, str, or float
            the value associated with this field
        datatype : int, str, or float type
            what type of data is expected for the value and default
        nullable : bool
            if true, the default will be used in the case of None for the value
        mutable : bool
            if false, the value can never be changed after creation
        default : int, str, or float
            the default value to use if value is None and nullable is True

        """
        if value is not None and not isinstance(value, datatype):
            msg = f"MetaField value and kind must match. Found kind={datatype} and value={type(value)}."
            raise TypeError(msg)
        if default is not None and not isinstance(default, datatype):
            msg = f"MetaField default and kind must match. Found kind={datatype} and default={type(default)}."
            raise TypeError(
                msg,
            )
        if len(keyword) > 8:
            msg = "Keywords must be 8 characters or shorter to comply with FITS"
            raise ValueError(msg)
        self._keyword = keyword
        self._comment = comment
        self._value = value
        self._datatype = datatype
        self.nullable = nullable
        self._mutable = mutable
        self._default = default

    @property
    def datatype(self) -> t.Any:
        """Get the data type."""
        return self._datatype

    @property
    def keyword(self) -> str:
        """Returns MetaField keyword."""
        return self._keyword

    @property
    def comment(self) -> str:
        """Returns MetaField comment."""
        return self._comment

    @property
    def value(self) -> ValueType:
        """Returns MetaField value."""
        return self._value

    @value.setter
    def value(self, value: ValueType) -> None:
        """Set value within MetaField object."""
        if not self._mutable:
            msg = "Cannot mutate this value because it is set to immutable."
            raise RuntimeError(msg)
        if isinstance(value, self._datatype) or value is None:
            self._value = value
        else:
            msg = f"Value of {self.keyword} was {type(value)} but must be {self._datatype}."
            raise TypeError(msg)

    @property
    def default(self) -> ValueType:
        """Get the default value."""
        return self._default

    @default.setter
    def default(self, default: ValueType) -> None:
        if isinstance(default, self._datatype) or default is None:
            self._default = default
        else:
            msg = f"Value was {type(default)} but must be {self._default}."
            raise TypeError(msg)

    def __eq__(self, other: MetaField) -> bool:
        """Check equality."""
        if not isinstance(other, MetaField):
            msg = f"MetaFields can only be compared to their own type, found {type(other)}."
            raise TypeError(msg)
        return (
            self._keyword == other._keyword
            and self._comment == other._comment
            and self._value == other._value
            and self._datatype == other._datatype
            and self.nullable == other.nullable
            and self._mutable == other._mutable
            and self._default == other._default
        )


class NormalizedMetadata(Mapping):
    """
    Represent Metadata consistently.

    The NormalizedMetadata object standardizes metadata and metadata access in the PUNCH pipeline. It does so by
    uniting the history and header fields while providing helpful accessors for commonly used formats of the metadata.

    Internally, the keys are always stored as FITS compliant upper-case strings. These are stored in sections.
    So the contents dictionary should have a key of a section title mapped to a dictionary of field keys mapped to
    MetaFields.
    """

    def __len__(self) -> int:
        """Return number of entry cards in NormalizedMetadata object."""
        return sum([len(section) for section in self._contents.values()])

    def __init__(
            self,
            contents: t.OrderedDict[str, t.OrderedDict[str, MetaField]],
            history: History | None = None,
            wcs_section_name: str = "World Coordinate System",
    ) -> None:
        """
        Create a Normalized Metadata. Also see `from_template` as that is often more helpful.

        Parameters
        ----------
        contents: OrderedDict[str, OrderedDict[str, MetaField]]
            contents of the meta information
        history: History
            history contents for this meta field
        wcs_section_name: str
            the section title for the WCS section to specially fill

        """
        self._contents = contents
        self._history = history if history is not None else History()
        self._wcs_section_name = wcs_section_name

    def __iter__(self) -> t.Iterator[t.Any]:
        """Iterate."""
        return self._contents.__iter__()

    def __eq__(self, other: NormalizedMetadata) -> bool:
        """Check equality."""
        if not isinstance(other, NormalizedMetadata):
            msg = f"Can only check equality between two NormalizedMetadata, found {type(other)}."
            raise TypeError(msg)
        return self._contents == other._contents and self._history == other._history

    def to_fits_header(self, wcs: WCS | None = None, write_celestial_wcs: bool = True) -> Header:  # noqa: C901
        """
        Convert a constructed NormalizedMetdata object to an Astropy FITS compliant header object.

        Returns
        -------
        Header
            Astropy FITS compliant header object

        """
        hdr = fits.Header()
        for section in self._contents:
            hdr.append(
                ("COMMENT", ("----- " + section + " ").ljust(72, "-")),
                end=True,
            )
            # for normal sections
            for field in self._contents[section].values():
                if field.value is not None:
                    value = field.value
                elif field.value is None and field.nullable:
                    value = field.default
                else:
                    msg = f"Value is null for {field.keyword} and no default is allowed."
                    raise RuntimeError(msg)
                hdr.append(
                    (
                        field.keyword,
                        value,
                        field.comment,
                    ),
                    end=True,
                )

            # add the special WCS section
            if section == self._wcs_section_name:
                if wcs is None:
                    msg = "WCS was expected but not provided."
                    raise MissingMetadataError(msg)
                if write_celestial_wcs:
                    wcses = {"": wcs, "A": calculate_celestial_wcs_from_helio(wcs, self.astropy_time, self.shape)}
                else:
                    wcses = {"": wcs}
                for key, this_wcs in wcses.items():
                    if this_wcs.has_distortion:
                        wcs_header = this_wcs.to_fits()[0].header
                        for required_key in REQUIRED_HEADER_KEYWORDS:
                            del wcs_header[required_key]
                    else:
                        wcs_header = this_wcs.to_header()
                    for card in wcs_header.cards:
                        if key == "" or (key != "" and card[0][-1].isnumeric() and card[0] not in DISTORTION_KEYWORDS):
                            hdr.append(
                                (
                                card[0] + key,
                                card[1],
                                card[2],
                                ),
                                end=True,
                            )

        # add the history section
        for entry in self.history:
            hdr["HISTORY"] = f"{entry.datetime: %Y-%m-%dT%H:%M:%S} => {entry.source} => {entry.comment}|"

        # fill in dynamic values
        if wcs is not None:
            geocentric = GCRS(obstime=self.astropy_time)
            p_angle = _sun_north_angle_to_z(geocentric)
            center_helio_coord = SkyCoord(
                wcs.wcs.crval[0] * u.deg,
                wcs.wcs.crval[1] * u.deg,
                frame=frames.Helioprojective,
                obstime=self.astropy_time,
                observer="earth",
            )
            hdr["RSUN_ARC"] = solar_angular_radius(center_helio_coord).value
            hdr["SOLAR_EP"] = p_angle.value
            hdr["CAR_ROT"] = float(sun.carrington_rotation_number(t=self.astropy_time))

        return hdr

    def delete_section(self, section_name: str) -> None:
        """
        Delete a section of NormalizedMetadata.

        Parameters
        ----------
        section_name : str
            the section to delete

        Returns
        -------
        None

        """
        if section_name in self._contents:
            del self._contents[section_name]
        else:
            msg = f"Section {section_name} was not found."
            raise MissingMetadataError(msg)


    @classmethod
    def from_fits_header(cls, h: Header) -> NormalizedMetadata:
        """
        Construct a normalized Metadata from a PUNCH FITS header.

        Parameters
        ----------
        h : Header
            a PUNCH FITS header from Astropy

        Returns
        -------
        NormalizedMetadata
            the corresponding NormalizedMetadata

        """
        if "TYPECODE" not in h:
            msg = "TYPECODE must a field of the header"
            raise MissingMetadataError(msg)
        if "OBSCODE" not in h:
            msg = "OBSCODE must be a field of the header"
            raise MissingMetadataError(msg)
        if "LEVEL" not in h:
            msg = "LEVEL must be a field of the header"
            raise MissingMetadataError(msg)

        type_code, obs_code, level = h["TYPECODE"], h["OBSCODE"], h["LEVEL"]

        m = NormalizedMetadata.load_template(type_code + obs_code, level)

        for k, v in h.items():
            if k not in ("COMMENT", "HISTORY", ""):
                if k not in m:
                    msg = f"Skipping unexpected key of {k} found in header for Level{level} {type_code + obs_code}."
                    warnings.warn(msg, ExtraMetadataWarning)
                else:
                    m[k] = v
        m.history = History.from_fits_header(h)

        return m

    @staticmethod
    def _match_product_code_in_level_spec(product_code: str, level_spec: dict) -> dict:
        """
        Parse the specified product code and level specification to find a corresponding set.

        Parameters
        ----------
        product_code
            Specified data product code
        level_spec
            Data product level specifications, loaded from `load_level_spec`

        Returns
        -------
        Dict
            Product code specification parsed from file

        """
        if product_code in level_spec["Products"]:
            return level_spec["Products"][product_code]
        else:  # noqa: RET505, okay structure
            type_code = product_code[:-1]
            found_type_codes = {pc[:-1] for pc in level_spec["Products"]}
            if type_code in found_type_codes:
                return level_spec["Products"][type_code + "?"]
            else:  # noqa: RET505, okay structure
                msg = f"Product code {product_code} not found in level_spec"
                raise RuntimeError(msg)

    @staticmethod
    def _load_template_files(
        omniheader_path: str, level: str, level_spec_path: str, spacecraft: str, spacecraft_def_path: str,
    ) -> tuple[dict, dict, dict]:
        """
        Load template files from specified locations.

        Parameters
        ----------
        omniheader_path
            Path to full omniheader specifications
        level
            Specified data product level
        level_spec_path
            Path to data product level specifications
        spacecraft
            Specified spacecraft code
        spacecraft_def_path
            Path to spacecraft specifications

        Returns
        -------
        Tuple
            Header specification entries

        """
        omniheader = load_omniheader(omniheader_path)
        spacecraft_def = load_spacecraft_def(spacecraft_def_path)
        if spacecraft not in spacecraft_def:
            msg = f"Spacecraft {spacecraft} not in spacecraft_def."
            raise RuntimeError(msg)

        if level is not None and level_spec_path is not None:
            msg = "Only specify the level or level_spec_path, not both."
            raise RuntimeError(msg)
        elif level is not None:  # noqa: RET506, fine structure
            level_spec_path = os.path.join(_ROOT, "data", f"Level{level}.yaml")
            level_spec = load_level_spec(level_spec_path)
        elif level_spec_path is not None:
            level_spec = load_level_spec(level_spec_path)
        else:
            msg = "Either level or level_spec_path must be defined. Found None for both."
            raise RuntimeError(msg)
        return omniheader, level_spec, spacecraft_def

    @staticmethod
    def _determine_omits_and_overrides(  # noqa: C901
        level_spec: dict,  # , not too complex
        product_def: dict,
    ) -> tuple[list[str], dict[str, str]]:
        """
        Read level specifications and product definitions and determines keywords to omit or overwrite.

        Parameters
        ----------
        level_spec
            Data product level specifications
        product_def
            Data product specifications

        Returns
        -------
        Tuple
            Keywords and values to omit and override

        """
        this_kinds = product_def["kinds"]
        omits, overrides = [], {}
        for section in level_spec["Level"]:
            if level_spec["Level"][section] is not None:
                if "omits" in level_spec["Level"][section]:
                    omits += level_spec["Level"][section]["omits"]
                if "overrides" in level_spec["Level"][section]:
                    for key, value in level_spec["Level"][section]["overrides"].items():
                        overrides[key] = value

        for kind in this_kinds:
            if kind not in level_spec["Kinds"]:
                msg = f"{kind} not found in level_spec."
                raise RuntimeError(msg)
            if "omits" in level_spec["Kinds"][kind]:
                omits += level_spec["Kinds"][kind]["omits"]
            if "overrides" in level_spec["Kinds"][kind]:
                for key, value in level_spec["Kinds"][kind]["overrides"].items():
                    overrides[key] = value

        if "omits" in product_def:
            omits += product_def["omits"]

        if "overrides" in product_def:
            for key, value in product_def["overrides"].items():
                overrides[key] = value

        return omits, overrides

    @classmethod
    def load_template(  # noqa: C901
        cls,
        product_code: str,
        level: str | None = None,
        level_spec_path: str | None = None,
        omniheader_path: str | None = None,
        spacecraft_def_path: str | None = None,
    ) -> NormalizedMetadata:
        """
        Given data product specification, loads relevant template files and constructs a NormalizedMetadata object.

        Parameters
        ----------
        product_code
            Specified data product code, a three character code like PM1
        level
            Specified data product level
        level_spec_path
            Path to data product level specifications
        omniheader_path
            Path to full omniheader specifications
        spacecraft_def_path
            Path to spacecraft specifications

        Returns
        -------
        NormalizedMetadata
            Constructed NormalizedMetadata object from template specifications

        """
        # load all needed files
        spacecraft = product_code[-1]
        omniheader, level_spec, spacecraft_def = NormalizedMetadata._load_template_files(
            omniheader_path, level, level_spec_path, spacecraft, spacecraft_def_path,
        )

        product_def = NormalizedMetadata._match_product_code_in_level_spec(product_code, level_spec)
        omits, overrides = NormalizedMetadata._determine_omits_and_overrides(level_spec, product_def)

        # construct the items to fill
        contents, history = OrderedDict(), History()

        # figure out the sections
        section_rows = np.where(omniheader["TYPE"] == "section")[0]
        section_titles = omniheader["VALUE"].iloc[section_rows]
        section_ids = omniheader["SECTION"].iloc[section_rows]

        # parse each section
        dtypes = {"str": str, "int": int, "float": float}
        for section_id, section_title in zip(section_ids, section_titles, strict=False):
            if section_title in level_spec["Level"]:
                contents[section_title] = OrderedDict()
                for i in np.where(omniheader["SECTION"] == section_id)[0][1:]:
                    e = omniheader.iloc[i]
                    if e["KEYWORD"] not in omits:
                        datatype = dtypes[e["DATATYPE"]]
                        value, default = e["VALUE"], e["DEFAULT"]
                        if e["KEYWORD"] in overrides:
                            value = overrides[e["KEYWORD"]]
                        try:
                            if datatype is str:
                                value = datatype(value)
                                value = value.format(**spacecraft_def[spacecraft])
                            elif (datatype is int) or (datatype is float):
                                value = datatype(value) if value != "" else None
                        except ValueError as err:
                            msg = f"Value was of the wrong type to parse for {e['KEYWORD']}"
                            raise RuntimeError(msg) from err

                        try:
                            if datatype is str:
                                default = datatype(default)
                                default = default.format(**spacecraft_def[spacecraft])
                            elif (datatype is int) or (datatype is float):
                                default = datatype(default) if default != "" else None
                        except ValueError as err:
                            msg = f"Default was of the wrong type to parse for {e['KEYWORD']}"
                            raise RuntimeError(msg) from err

                        contents[section_title][e["KEYWORD"]] = MetaField(
                            e["KEYWORD"],
                            e["COMMENT"].format(**spacecraft_def[spacecraft]),
                            value,
                            datatype,
                            e["NULLABLE"],
                            e["MUTABLE"],
                            default,
                        )
        return cls(contents, history)

    @property
    def sections(self) -> list[str]:
        """Returns header sections."""
        return list(self._contents.keys())

    @property
    def fits_keys(self) -> list[str]:
        """Returns fits keys in header template."""

        def flatten(xss: list) -> list:
            return [x for xs in xss for x in xs]

        return flatten([list(self._contents[section_name].keys()) for section_name in self._contents])

    @property
    def history(self) -> History:
        """Returns header history."""
        return self._history

    @history.setter
    def history(self, history: History) -> None:
        self._history = history

    @staticmethod
    def _validate_key_is_str(key: str) -> None:
        """
        Validate that the provided key is a valid header keyword string.

        Parameters
        ----------
        key
            Header key string

        Returns
        -------
        None

        """
        if not isinstance(key, str):
            msg = f"Keys for NormalizedMetadata must be strings. You provided {type(key)}."
            raise TypeError(msg)
        if len(key) > 8:
            msg = f"Keys must be <= 8 characters long, received {key}"
            raise ValueError(msg)

    def __setitem__(self, key: str, value: t.Any) -> None:
        """
        Set specified pair of keyword and value in the NormalizedMetadata object.

        Parameters
        ----------
        key
            Header key string
        value
            Header value

        Returns
        -------
        None

        """
        for section_name, section in self._contents.items():
            if key in section:
                self._contents[section_name][key.upper()].value = value
                return

        # reaching here means we haven't returned
        msg = f"MetaField with key={key} not found."
        raise RuntimeError(msg)

    def __getitem__(self, key: str | tuple[str, int]) -> t.Any:
        """
        Get specified keyword from NormalizedMetadata object.

        Parameters
        ----------
        key : str | tuple
            Header key string

        Returns
        -------
        t.Any
            Returned header value

        """
        if isinstance(key, tuple):
            key, i = key
        self._validate_key_is_str(key)

        for section_name, section in self._contents.items():
            if key in section:
                return self._contents[section_name][key.upper()]

        # reaching here means we haven't returned
        msg = f"MetaField with key={key} not found."
        raise RuntimeError(msg)

    def __delitem__(self, key: str) -> None:
        """
        Delete specified keyword entry from the NormalizedMetadata object.

        Parameters
        ----------
        key
            Header key string

        Returns
        -------
        None

        """
        self._validate_key_is_str(key)
        for section_name, section in self._contents.items():
            if key in section:
                del self._contents[section_name][key.upper()]
                return

        # reaching here means we haven't returned
        msg = f"MetaField with key={key} not found."
        raise RuntimeError(msg)

    def __contains__(self, key: str) -> bool:
        """
        Determine if the specified keyword is contained within the NormalizedMetadata object.

        Parameters
        ----------
        key
            Header key string

        Returns
        -------
        Boolean
            Value indicating if the specified keyword is contained within the NormalizedMetadata object

        """
        return any(key in section for section in self._contents.values())

    @property
    def product_level(self) -> int:
        """Returns data product level if indicated in metadata."""
        if "LEVEL" not in self:
            msg = "LEVEL is missing from the metadata."
            raise MissingMetadataError(msg)
        return self["LEVEL"].value

    @property
    def datetime(self) -> datetime:
        """Returns a datetime representation of the 'DATE-OBS' header keyword if indicated in metadata."""
        if "DATE-OBS" not in self:
            msg = "DATE-OBS is missing from the metadata."
            raise MissingMetadataError(msg)
        return parse_datetime(self["DATE-OBS"].value)

    @property
    def shape(self) -> tuple:
        """Get the data shape in array order."""
        return tuple([self[f"NAXIS{i}"].value for i in range(self["NAXIS"].value, 0, -1)])

    @property
    def astropy_time(self) -> Time:
        """Get the date-obs as an astropy Time object."""
        return Time(self.datetime)
