from __future__ import annotations

import typing as t
import os.path
from datetime import datetime
from collections import OrderedDict, namedtuple
from collections.abc import Mapping

import astropy.units as u
import astropy.wcs.wcsapi
import matplotlib as mpl
import numpy as np
import pandas as pd
import sunpy.map
import yaml
from astropy.coordinates import GCRS, EarthLocation, SkyCoord, StokesSymbol, custom_stokes_symbol_mapping
from astropy.io import fits
from astropy.io.fits import Header
from astropy.nddata import StdDevUncertainty
from astropy.time import Time
from astropy.wcs import WCS
from dateutil.parser import parse as parse_datetime
from ndcube import NDCube
from sunpy.coordinates import frames, sun
from sunpy.coordinates.sun import _sun_north_angle_to_z
from sunpy.map import solar_angular_radius

from punchbowl.exceptions import MissingMetadataError

_ROOT = os.path.abspath(os.path.dirname(__file__))

PUNCH_STOKES_MAPPING = custom_stokes_symbol_mapping({10: StokesSymbol("pB", "polarized brightness"),
                                                     11: StokesSymbol("B", "total brightness")})


def get_data_path(path: str) -> str:
    """Return root data path given the filename requested."""
    return os.path.join(_ROOT, "data", path)


def load_omniheader(path: str | None = None) -> pd.DataFrame:
    """Load full metadata specifications."""
    if path is None:
        path = get_data_path("omniheader.csv")
    return pd.read_csv(path, na_filter=False)


def load_level_spec(path: str) -> dict[str, t.Any]:
    """Load data product metadata specifications."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_trefoil_wcs() -> astropy.wcs.WCS:
    """Load Level 2 trefoil world coordinate system and shape."""
    trefoil_wcs = WCS(get_data_path("trefoil_hdr.fits"))
    trefoil_wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"  # TODO: figure out why this is necessary, seems like a bug
    trefoil_shape = (4096, 4096)
    return trefoil_wcs, trefoil_shape


def load_spacecraft_def(path: str | None = None) -> dict[str, t.Any]:
    """
    Load spacecraft metadata specifications.

    If path is None, then it loads a default from the package.
    """
    if path is None:
        path = get_data_path("spacecraft.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def extract_crota_from_wcs(wcs: WCS) -> tuple[float, float]:
    """Extract CROTA from a WCS."""
    return np.arctan2(wcs.wcs.pc[1, 0], wcs.wcs.pc[0, 0]) * u.rad


def calculate_helio_wcs_from_celestial(wcs_celestial: WCS, date_obs: datetime, data_shape: tuple[int]) -> WCS:
    """Calculate the helio WCS from a celestial WCS."""
    is_3d = len(data_shape) == 3

    # we're at the center of the Earth
    test_loc = EarthLocation.from_geocentric(0, 0, 0, unit=u.m)
    test_gcrs = SkyCoord(test_loc.get_gcrs(date_obs))

    # follow the SunPy tutorial from here
    # https://docs.sunpy.org/en/stable/generated/gallery/units_and_coordinates/radec_to_hpc_map.html#sphx-glr-generated-gallery-units-and-coordinates-radec-to-hpc-map-py
    reference_coord = SkyCoord(
        wcs_celestial.wcs.crval[0] * u.Unit(wcs_celestial.wcs.cunit[0]),
        wcs_celestial.wcs.crval[1] * u.Unit(wcs_celestial.wcs.cunit[1]),
        frame="gcrs",
        obstime=date_obs,
        obsgeoloc=test_gcrs.cartesian,
        obsgeovel=test_gcrs.velocity.to_cartesian(),
        distance=test_gcrs.hcrs.distance,
    )

    reference_coord_arcsec = reference_coord.transform_to(frames.Helioprojective(observer=test_gcrs))

    cdelt1 = (np.abs(wcs_celestial.wcs.cdelt[0]) * u.deg).to(u.arcsec)
    cdelt2 = (np.abs(wcs_celestial.wcs.cdelt[1]) * u.deg).to(u.arcsec)

    geocentric = GCRS(obstime=date_obs)
    p_angle = _sun_north_angle_to_z(geocentric)

    crota = extract_crota_from_wcs(wcs_celestial)

    new_header = sunpy.map.make_fitswcs_header(
        data_shape[1:] if is_3d else data_shape,
        reference_coord_arcsec,
        reference_pixel=u.Quantity(
            [wcs_celestial.wcs.crpix[0] - 1, wcs_celestial.wcs.crpix[1] - 1] * u.pixel,
        ),
        scale=u.Quantity([cdelt1, cdelt2] * u.arcsec / u.pix),
        rotation_angle=-p_angle - crota,
        observatory="PUNCH",
        projection_code=wcs_celestial.wcs.ctype[0][-3:],
    )

    wcs_helio = WCS(new_header)

    if is_3d:
        wcs_helio = astropy.wcs.utils.add_stokes_axis_to_wcs(wcs_helio, 2)

    return wcs_helio, p_angle


HistoryEntry = namedtuple("HistoryEntry", "datetime, source, comment")


class History:
    """Representation of the history of edits done to a PUNCHData object."""

    def __init__(self) -> None:
        """Create blank history."""
        self._entries: list[HistoryEntry] = []

    def add_entry(self, entry: HistoryEntry) -> None:
        """
        Add an entry to the History log.

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
        """
        Add a new history entry at the current time.

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
        Clear all the history entries so the History is blank.

        Returns
        -------
        None

        """
        self._entries = []

    def __getitem__(self, index: int) -> HistoryEntry:
        """
        Given an index, return the requested HistoryEntry.

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
        Get the most recent HistoryEntry, i.e. the youngest.

        Returns
        -------
        HistoryEntry
            returns HistoryEntry that is the youngest

        """
        return self._entries[-1]

    def __len__(self) -> int:
        """Get length."""
        return len(self._entries)

    def __str__(self) -> str:
        """
        Format a string combining all the history entries.

        Returns
        -------
        str
            a combined record of the history entries

        """
        return "\n".join([f"{e.datetime}: {e.source}: {e.comment}" for e in self._entries])

    def __iter__(self) -> History:
        """Iterate."""
        self.current_index = 0
        return self

    def __next__(self) -> HistoryEntry:
        """Get next."""
        if self.current_index >= len(self):
            raise StopIteration
        entry = self._entries[self.current_index]
        self.current_index += 1
        return entry

    def __eq__(self, other: History) -> bool:
        """Check equality of two History objects."""
        if not isinstance(other, History):
            msg = f"Can only check equality between two history objects, found History and {type(other)}"
            raise TypeError(msg)
        return self._entries == other._entries

    @classmethod
    def from_fits_header(cls, head: Header) -> History:
        """
        Construct a history from a FITS header.

        Parameters
        ----------
        head : Header
            a FITS header to read from

        Returns
        -------
        History
            the history derived from a given FITS header

        """
        if "HISTORY" not in head:
            out = cls()
        else:
            out = cls()
            for row in head["HISTORY"][1:]:
                dt, source, comment = row.split(" => ")
                dt = parse_datetime(dt)
                out.add_entry(HistoryEntry(dt, source, comment))
        return out


ValueType = int | str | float


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
        self, contents: t.OrderedDict[str, t.OrderedDict[str, MetaField]], history: History | None = None,
    ) -> None:
        """
        Create a Normalized Metadata. Also see `from_template` as that is often more helpful.

        Parameters
        ----------
        contents: OrderedDict[str, OrderedDict[str, MetaField]]
            contents of the meta information
        history: History
            history contents for this meta field

        """
        self._contents = contents
        self._history = history if history is not None else History()

    def __iter__(self) -> t.Iterator[t.Any]:
        """Iterate."""
        return self._contents.__iter__()

    def __eq__(self, other: NormalizedMetadata) -> bool:
        """Check equality."""
        if not isinstance(other, NormalizedMetadata):
            msg = f"Can only check equality between two NormalizedMetadata, found {type(other)}."
            raise TypeError(msg)
        return self._contents == other._contents and self._history == other._history

    def to_fits_header(self) -> Header:
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
        for entry in self.history:
            hdr["HISTORY"] = f"{entry.datetime} => {entry.source} => {entry.comment}"
        return hdr

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
            raise RuntimeError(msg)
        if "OBSCODE" not in h:
            msg = "OBSCODE must be a field of the header"
            raise RuntimeError(msg)
        if "LEVEL" not in h:
            msg = "LEVEL must be a field of the header"
            raise RuntimeError(msg)

        type_code, obs_code, level = h["TYPECODE"], h["OBSCODE"], h["LEVEL"]

        m = NormalizedMetadata.load_template(type_code + obs_code, level)

        for k, v in h.items():
            if k not in ("COMMENT", "HISTORY", ""):
                if k not in m:
                    msg = f"Unexpected key of {k} found in header for Level{level} {type_code + obs_code} type meta."
                    raise RuntimeError(
                        msg,
                    )
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
            level_spec_path = get_data_path(f"Level{level}.yaml")
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
        self._validate_key_is_str(key)
        for section_name, section in self._contents.items():
            if key in section:
                self._contents[section_name][key.upper()].value = value
                return

        # reaching here means we haven't returned
        msg = f"MetaField with key={key} not found."
        raise RuntimeError(msg)

    def __getitem__(self, key: str) -> t.Any:
        """
        Get specified keyword from NormalizedMetadata object.

        Parameters
        ----------
        key
            Header key string

        Returns
        -------
        t.Any
            Returned header value

        """
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
        self._validate_key_is_str(key)
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
            uncertainty = (secondary_hdu.data / 255).astype(np.float32)
            if (uncertainty.min() < 0) or (uncertainty.max() > 1):
                msg = "Uncertainty array in file outside of expected range (0-1)."
                raise ValueError(msg)
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


def get_base_file_name(cube: NDCube) -> str:
    """Determine the base file name without file type extension."""
    obscode = cube.meta["OBSCODE"].value
    file_level = cube.meta["LEVEL"].value
    type_code = cube.meta["TYPECODE"].value
    date_string = cube.meta.datetime.strftime("%Y%m%d%H%M%S")
    # TODO: include version number
    return "PUNCH_L" + file_level + "_" + type_code + obscode + "_" + date_string


def write_ndcube_to_fits(cube: NDCube,
                         filename: str,
                         overwrite: bool = True,
                         skip_wcs_conversion: bool = False) -> None:
    """Write an NDCube as a FITS file."""
    if filename.endswith(".fits"):
        _write_fits(cube, filename, overwrite=overwrite, skip_wcs_conversion=skip_wcs_conversion)
    elif filename.endswith((".png", ".jpg", ".jpeg")):
        _write_ql(cube, filename, overwrite=overwrite)
    else:
        msg = (
            "Filename must have a valid file extension (.fits, .png, .jpg, .jpeg). "
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

    helio_wcs_header = helio_wcs.to_header()

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


def _write_fits(cube: NDCube, filename: str, overwrite: bool = True, skip_wcs_conversion: bool = False) -> None:
    _update_statistics(cube)

    header = cube.meta.to_fits_header()

    # update the header with the WCS
    # TODO: explain the skip_wcs_conversion step
    wcs_header = cube.wcs.to_header() if skip_wcs_conversion else construct_wcs_header_fields(cube)
    for key, value in wcs_header.items():
        if key in header:
            header[key] = (cube.meta[key].datatype)(value)
            cube.meta[key] = (cube.meta[key].datatype)(value)

    hdul_list = []
    hdu_dummy = fits.PrimaryHDU()
    hdul_list.append(hdu_dummy)

    hdu_data = fits.CompImageHDU(data=cube.data, header=header, name="Primary data array")
    hdul_list.append(hdu_data)

    if cube.uncertainty is not None:
        if (cube.uncertainty.array.min() < 0) or (cube.uncertainty.array.max() > 1):
            msg = "Uncertainty array outside of expected range (0-1)."
            raise ValueError(msg)
        scaled_uncertainty = (cube.uncertainty.array * 255).astype(np.uint8)
        hdu_uncertainty = fits.CompImageHDU(data=scaled_uncertainty, name="Uncertainty array")
        # write WCS to uncertainty header
        for key, value in wcs_header.items():
            hdu_uncertainty.header[key] = value
        # Save as an 8-bit unsigned integer
        hdul_list.append(hdu_uncertainty)

    hdul = fits.HDUList(hdul_list)

    hdul.writeto(filename, overwrite=overwrite, checksum=True)


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
