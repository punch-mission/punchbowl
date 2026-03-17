POLS = ["R", "M", "Z", "P"]
PREFIXES = ["X", "Y", "S", "T", "G"]
L0_1_CODES = ["CR", "PP", "PM", "PZ", "DK", "DS", "DY", "FQ", "QR", "RC", "RM", "RZ", "RP"]
L2_3_CODES = ["PT", "CT", "PI", "CI", "PF", "CF", "PA", "CA", "CN", "CQ"]
CODES = L0_1_CODES + [pre + c for c in POLS for pre in PREFIXES] + L2_3_CODES
ALPHABET = "abcdefghijklmnopqrstuvwxyz"

from itertools import product

from sunpy.net.dataretriever.client import GenericClient, QueryResponse
from sunpy.net.scraper import Scraper
from sunpy.time import TimeRange


class PUNCHClient(GenericClient):
    pattern = ("https://umbra.nascom.nasa.gov/punch/{Level}/{ProductCode}{Instrument}/{{year:4d}}/{{month:2d}}/"
               "{{day:2d}}/PUNCH_L{Level}_{ProductCode}{Instrument}_{{year:4d}}{{month:2d}}{{day:2d}}{{hour:2d}}"
               "{{minute:2d}}{{second:2d}}_v{DataVersion}.fits")

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
        }
        return adict

    instr_replacements = {"wfi-1": 1, "wfi-2": 2, "wfi-3": 3, "nfi-4": 4, "m": "M"}

    def search(self, *args, **kwargs):
        matchdict = self._get_match_dict(*args, **kwargs)
        req_codes = matchdict.get("ProductCode")
        req_instrs = matchdict.get("Instrument")
        req_levels = matchdict.get("Level")
        req_versions = matchdict.get("DataVersion")

        metalist = []
        for code, instr, level, dversion in product(req_codes, req_instrs, req_levels, req_versions):
            code = code.upper()
            url_instr = self.instr_replacements[instr]
            fdict = {"ProductCode": code, "Instrument": url_instr, "Level": level, "DataVersion": dversion}

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
