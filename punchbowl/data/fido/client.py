from itertools import product

from sunpy.net import attrs as a
from sunpy.net.attr import SimpleAttr
from sunpy.net.dataretriever.client import GenericClient, QueryResponse
from sunpy.net.scraper import Scraper
from sunpy.time import TimeRange

from .attrs import FileType

POLS = ["R", "M", "Z", "P"]
PREFIXES = ["X", "Y", "S", "T", "G"]
L0_1_CODES = ["CR", "PP", "PM", "PZ", "DK", "DS", "DY", "FQ", "QR", "RC", "RM", "RZ", "RP"]
L2_3_CODES = ["PT", "CT", "PI", "CI", "PF", "CF", "PA", "CA", "CN", "CQ"]
CODES = L0_1_CODES + [pre + c for c in POLS for pre in PREFIXES] + L2_3_CODES
ALPHABET = "abcdefghijklmnopqrstuvwxyz"


class PUNCHClient(GenericClient):
    pattern = ("https://umbra.nascom.nasa.gov/punch/{Level}/{ProductCode}{Instrument}/{{year:4d}}/{{month:2d}}/"
               "{{day:2d}}/PUNCH_L{Level}_{ProductCode}{Instrument}_{{year:4d}}{{month:2d}}{{day:2d}}{{hour:2d}}"
               "{{minute:2d}}{{second:2d}}_v{DataVersion}.{FileType}")

    @classmethod
    def _attrs_module(cls):
        return "punch", "punchbowl.data.fido.attrs"

    @classmethod
    def register_values(cls):
        from sunpy.net import attrs
        adict = {
            attrs.Level: [("0", "L0"),
                          ("1", "L1"),
                          ("2", "L2"),
                          ("3", "L3"),
                          ("Q", "LQ")],
            attrs.Instrument: [("WFI-1", "Wide Field Imager 1"),
                               ("WFI-2", "Wide Field Imager 2"),
                               ("WFI-3", "Wide Field Imager 3"),
                               ("NFI-4", "Narrow Field Imager"),
                               ("M", "PUNCH Mosaic") ],
            attrs.punch.ProductCode: [(code, code) for code in CODES],
            attrs.punch.DataVersion: [(f"{v}{subv}", f"{v}{subv}") for subv in ALPHABET for v in range(2)],
            attrs.Source: [("PUNCH", "Polarimeter to UNify the Corona and Heliosphere")],
            attrs.Provider: [("SwRI", "Southwest Research Institute")],
            attrs.punch.FileType: [("fits", "FITS file"), ("jp2", "Quick-look JPEG2000")],
        }
        return adict

    instr_replacements = {"wfi-1": 1, "wfi-2": 2, "wfi-3": 3, "nfi-4": 4, "m": "M"}

    def _get_match_dict(cls, *args, **kwargs):
        """
        Constructs a dictionary using the query and registered Attrs that represents
        all possible values of the extracted metadata for files that matches the query.
        The returned dictionary is used to validate the metadata of searched files
        in :func:`~sunpy.net.scraper.Scraper._extract_files_meta`.

        Parameters
        ----------
        \\*args: `tuple`
            `sunpy.net.attrs` objects representing the query.
        \\*\\*kwargs: `dict`
            Any extra keywords to refine the search.

        Returns
        -------
        matchdict: `dict`
            A dictionary having a `list` of all possible Attr values
            corresponding to an Attr.

        """
        regattrs_dict = cls.register_values()
        matchdict = {}
        for i in regattrs_dict.keys():
            attrname = i.__name__
            # only Attr values that are subclas of Simple Attr are stored as list in matchdict
            # since complex attrs like Range can't be compared with string matching.
            if i is FileType:
                matchdict[attrname] = ["fits"]
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
                raise ValueError(
                    f"GenericClient can not add {elem.__class__.__name__} to the rowdict dictionary to pass to the Client.")
        return matchdict

    def search(self, *args, **kwargs):
        for arg in args:
            print(arg)
        for k, v in kwargs.items():
            print(k, v)
        matchdict = self._get_match_dict(*args, **kwargs)
        print(matchdict)
        req_codes = matchdict.get("ProductCode")
        req_instrs = matchdict.get("Instrument")
        req_levels = matchdict.get("Level")
        req_versions = matchdict.get("DataVersion")
        req_file_types = matchdict.get("FileType")

        metalist = []
        for code, instr, level, dversion, file_type in product(
                req_codes, req_instrs, req_levels, req_versions, req_file_types):
            code = code.upper()
            url_instr = self.instr_replacements[instr]
            fdict = {"ProductCode": code, "Instrument": url_instr, "Level": level, "DataVersion": dversion,
                     "FileType": file_type}

            urlpattern = self.pattern.format(**fdict)
            urlpattern = urlpattern.replace("{", "{{").replace("}", "}}")
            scraper = Scraper(format=urlpattern)
            tr = TimeRange(matchdict["Start Time"], matchdict["End Time"])
            filesmeta = scraper._extract_files_meta(tr)
            for i in filesmeta:
                rowdict = self.post_search_hook(i, matchdict)
                rowdict["ProductCode"] = code
                rowdict["Instrument"] = instr.upper()
                rowdict["DataVersion"] = dversion.lower()
                rowdict["Provider"] = "SwRI"
                rowdict["Level"] = level.upper()
                metalist.append(rowdict)
        return QueryResponse(metalist, client=self)
