from itertools import product

from sunpy.net import attrs as a
from sunpy.net.attr import SimpleAttr
from sunpy.net.dataretriever.client import GenericClient, QueryResponse
from sunpy.net.scraper import Scraper
from sunpy.time import TimeRange

from .attrs import DataVersion, FileType

POLS = ["R", "M", "Z", "P"]
PREFIXES = ["X", "Y", "S", "T", "G"]
L0_1_CODES = ["CR", "PP", "PM", "PZ", "DK", "DS", "DY", "FQ", "QR", "RC", "RM", "RZ", "RP"]
L2_3_CODES = ["PT", "CT", "PI", "CI", "PF", "CF", "PA", "CA", "CN", "CQ"]
CODES = L0_1_CODES + [pre + c for c in POLS for pre in PREFIXES] + L2_3_CODES
ALPHABET = "abcdefghijklmnopqrstuvwxyz"


class PUNCHClient(GenericClient):
    """Fido client for fetching PUNCH data from the SDAC."""

    pattern = ("https://umbra.nascom.nasa.gov/punch/{Level}/{ProductCode}{Instrument}/{{year:4d}}/{{month:2d}}/"
               "{{day:2d}}/PUNCH_L{Level}_{ProductCode}{Instrument}_{{year:4d}}{{month:2d}}{{day:2d}}{{hour:2d}}"
               "{{minute:2d}}{{second:2d}}_v{{DataVersion}}.{FileType}")

    @classmethod
    def _attrs_module(cls) -> tuple[str, str]:
        """Tell Fido where our custom attributes are."""
        return "punch", "punchbowl.data.fido.attrs"

    @classmethod
    def register_values(cls) -> dict:
        """Register supported attrs with Fido."""
        return {
            a.Level: [("0", "L0"),
                          ("1", "L1"),
                          ("2", "L2"),
                          ("3", "L3"),
                          ("Q", "LQ")],
            a.Instrument: [("WFI-1", "Wide Field Imager 1"),
                               ("WFI-2", "Wide Field Imager 2"),
                               ("WFI-3", "Wide Field Imager 3"),
                               ("NFI-4", "Narrow Field Imager"),
                               ("M", "PUNCH Mosaic") ],
            a.punch.ProductCode: [(code, code) for code in CODES],
            a.punch.DataVersion: [("newest", "Newest available")]
                                 + [(f"{v}{subv}", f"{v}{subv}") for subv in ALPHABET for v in range(2)],
            a.Source: [("PUNCH", "Polarimeter to UNify the Corona and Heliosphere")],
            a.Provider: [("SwRI", "Southwest Research Institute")],
            a.punch.FileType: [("fits", "FITS file"), ("jp2", "Quick-look JPEG2000")],
        }

    @classmethod
    def _get_match_dict(cls, *args: tuple, **kwargs: dict) -> dict: # noqa: ARG003
        """
        Override of class method to support default value for FileType and DataVersion.

        Constructs a dictionary using the query and registered Attrs that represents
        all possible values of the extracted metadata for files that matches the query.
        The returned dictionary is used to validate the metadata of searched files
        in :func:`~sunpy.net.scraper.Scraper._extract_files_meta`.

        Parameters
        ----------
        args: `tuple`
            `sunpy.net.attrs` objects representing the query.
        kwargs: `dict`
            Any extra keywords to refine the search.

        Returns
        -------
        matchdict: `dict`
            A dictionary having a `list` of all possible Attr values
            corresponding to an Attr.

        """
        regattrs_dict = cls.register_values()
        matchdict = {}
        for i in regattrs_dict:
            attrname = i.__name__
            # only Attr values that are subclas of Simple Attr are stored as list in matchdict
            # since complex attrs like Range can't be compared with string matching.

            # HERE is where we deviate from the base class method, setting single default values for these two fields.
            if i is FileType:
                matchdict[attrname] = ["fits"]
            elif i is DataVersion:
                matchdict[attrname] = ["newest"]
            elif issubclass(i, SimpleAttr):
                matchdict[attrname] = []
                for val, _ in regattrs_dict[i]:
                    matchdict[attrname].append(val)
        for elem in args:
            if isinstance(elem, a.Time):
                matchdict["Start Time"] = elem.start
                matchdict["End Time"] = elem.end
            elif hasattr(elem, "value"):
                matchdict[elem.__class__.__name__] = [str(elem.value).lower()]
            elif isinstance(elem, a.Wavelength):
                matchdict["Wavelength"] = elem
            else:
                raise ValueError(f"GenericClient can not add {elem.__class__.__name__} to the rowdict dictionary to "
                                 f"pass to the Client.")
        return matchdict

    def search(self, *args: tuple, **kwargs: dict) -> QueryResponse:
        """Override of base class method to run the search, handling PUNCH attrs."""
        # matchdict will contain all the attributes we support querying. For each attribute, we'll have the
        # user-selected value if set, or the default if not. For some keys, the 'default' is all possible values.
        matchdict = self._get_match_dict(*args, **kwargs)
        req_codes = matchdict.get("ProductCode")
        req_instrs = matchdict.get("Instrument")
        req_levels = matchdict.get("Level")
        req_versions = matchdict.get("DataVersion")
        req_file_types = matchdict.get("FileType")

        instr_replacements = {"wfi-1": 1, "wfi-2": 2, "wfi-3": 3, "nfi-4": 4, "m": "M"}

        metalist = []
        # For every possible combination of requested values, we have to build and scrape a URL.
        for code, instr, level, dversion, file_type in product(
                req_codes, req_instrs, req_levels, req_versions, req_file_types):
            code = code.upper() # noqa: PLW2901
            url_instr = instr_replacements[instr]
            fdict = {"ProductCode": code, "Instrument": url_instr, "Level": level,
                     "FileType": file_type}

            # Scraper can only handle dates as "variables" in the url's directory path, so we have to fill in level
            # and product code. Scraper will handle day/month/year, and "wildcards" in the filename part of the URL
            # will be matched and the values extracted.
            urlpattern = self.pattern.format(**fdict)

            scraper = Scraper(format=urlpattern)
            tr = TimeRange(matchdict["Start Time"], matchdict["End Time"])
            filesmeta = scraper._extract_files_meta(tr) # noqa: SLF001
            # filesmeta will contain each file matching the pattern, as well as the matched values for all the "{{
            # Value}}" fields in the url pattern.

            if dversion == "newest":
                newest_file_by_date = {}
                for row in filesmeta:
                    # For each timestamp, we'll track the highest data version we see
                    tstamp = (row["year"], row["month"], row["day"], row["hour"], row["minute"], row["second"])
                    entry = (row["DataVersion"], row)
                    if tstamp in newest_file_by_date:
                        if row["DataVersion"] > newest_file_by_date[tstamp][0]:
                            newest_file_by_date[tstamp] = entry
                    else:
                        newest_file_by_date[tstamp] = entry
                # Now we'll extract the newest versions
                filesmeta = []
                for key in sorted(newest_file_by_date.keys()):
                    filesmeta.append(newest_file_by_date[key][1])

            for row in filesmeta:
                rowdict = self.post_search_hook(row, matchdict)
                # For the fields that we pasted into the URL pattern, we now have to put the corresponding values in
                # the table of files. We can also massage the values a bit.
                rowdict["ProductCode"] = code
                rowdict["Instrument"] = instr.upper()
                rowdict["DataVersion"] = row["DataVersion"].lower()
                rowdict["Provider"] = "SwRI"
                rowdict["Level"] = level.upper()
                rowdict["FileType"] = file_type
                metalist.append(rowdict)
        return QueryResponse(metalist, client=self)
