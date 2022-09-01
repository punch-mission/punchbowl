from __future__ import annotations
from collections import namedtuple
from datetime import datetime
import astropy.units as u
import matplotlib
from typing import Union, List, Dict, Any
import numpy as np
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from dateutil.parser import parse as parse_datetime
from ndcube import NDCube
import pandas

HistoryEntry = namedtuple("HistoryEntry", "datetime, source, comment")


class History:
    """Representation of the history of edits done to a PUNCHData object
    """
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

        """
        return self._entries[index]

    def most_recent(self) -> HistoryEntry:
        """
        Gets the most recent HistoryEntry, i.e. the youngest

        Returns
        -------
        HistoryEntry that is the youngest
        """
        return self._entries[-1]

    def __len__(self) -> int:
        """
        Returns
        -------
        int : the number of history entries
        """
        return len(self._entries)

    def __str__(self) -> str:
        """
        Formats a string combining all the history entries

        Returns
        -------
        str : a combined record of the history entries
        """
        return "\n".join([f"{e.datetime}: {e.source}: {e.comment}" for e in self._entries])


class PUNCHCalibration:
    """
    This will be inherited and developed as the various calibration objects, e.g. the quartic fit coefficients, are
    developed. This will be an abstract base class for all of them.
    """
    pass

class HeaderTemplate:
    """PUNCH data object header template
    Class to generate a PUNCH data object header template, along with associated methods.

    - TODO - Method to take in a data object - populate missing keywords, reorder, validate, etc
    (flag option for warnings for unpopulated keywords)
    (more general flag to supress non critical warnings)
    (custom warnings to supress particular types?)

    """

    def __init__(self, header_obj: Union[str, dict, None]):
        """Initialize the PUNCHDataHeader object with either a template file path,
        a dictionary object, or None.

        Parameters
        ----------
        header_obj
            input header object

        Returns
        ----------
        None

        """
        if isinstance(header_obj, str):
            self = self.load(header_obj)
        elif isinstance(header_obj, dict):
            self = fits.Header(header_obj)
        elif header_obj is None:
            #self = fits.Header({'SIMPLE': True})
            self = fits.Header()
        else:
            raise Exception("Please specify a template file path, a header dictionary, or None")

        self._history = History()

    @classmethod
    def load(cls, file:str) -> fits.Header:
        """
        Loads an input template file to generate a header object.

        Parameters
        ----------
        file
            input header template file

        Returns
        -------
        hdr
            astropy.io.fits.Header header object

        """

        if file.endswith('.txt'):
            hdr = cls._load_text(file)
        elif file.endswith('.tsv'):
            hdr = cls._load_tsv(file)
        elif file.endswith('.csv'):
            hdr = cls._load_csv(file)
        else:
            raise Exception('Specify a valid header template file extension (.txt, .tsv, .csv)')

        return hdr

    @staticmethod
    def _load_text(file: str) -> fits.Header:
        """
        Parses an input human-readable FITS header template and generates an astropy.io.fits.header
        FITS compliant header object.

        Parameters
        ----------
        file
            input header template file

        Returns
        -------
        hdr
            astropy.io.fits.header header object

        """

        with open(file, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if len(line) > 81:
                raise Exception('Header contains more than FITS standard specified 80 characters per line: ' + line)
            if len(line) < 81:
                 lines[i] = "{:<80}".format(line)


        reformatted_header = "".join([line[:-1] for line in lines])
        new_size = np.ceil(len(reformatted_header) / 2880).astype(int) * 2880
        num_spaces = new_size - len(reformatted_header)
        reformatted_header = reformatted_header + " " * num_spaces

        hdr = fits.header.Header.fromstring(reformatted_header)

        return hdr

    @staticmethod
    def _load_tsv(file: str) -> fits.Header:
        """
        Parses an input template header tab separated value (TSV) file to generate an astropy header object.

        Parameters
        ----------
        file
            input filename

        Returns
        -------
        hdr
            astropy.io.fits.header header object

        """

        hdr = fits.Header()

        with open(file, 'r') as csv_file:
            lines = csv_file.readlines()

        lines = [line[:-1] for line in lines]

        for line in lines[1:]:
            card = line.split('\t')

            if card[0] == 'section':
                if len(card[3]) > 72 : Warning("Section text exceeds 80 characters, EXTEND will be used.")
                hdr.append(('COMMENT', ('----- ' + card[3] + ' ').ljust(72,'-')), end=True)

            elif card[0] == 'comment':
                hdr.append(('COMMENT', card[2]), end=True)

            elif card[0] == 'keyword':
                if len(card[2]) + len(card[3]) > 72:
                    Warning("Section text exceeds 80 characters, EXTEND will be used.")

                if card[4] == 'str':
                    hdr.append((card[1], card[2], card[3]), end=True)
                elif card[4] == 'int':
                    hdr.append((card[1], int(card[2]), card[3]), end=True)
                elif card[4] == 'float':
                    hdr.append((card[1], np.float(card[2]), card[3]), end=True)

        return hdr

    @staticmethod
    def _load_csv(file: str) -> fits.Header:
        """
        Parses an input template header comma separated value (CSV) file to generate an astropy header object.

        Parameters
        ----------
        file
            input filename

        Returns
        -------
        hdr
            astropy.io.fits.header header object


        """

        hdr = fits.Header()

        template = pandas.read_csv(file, keep_default_na=False)

        for li in np.arange(len(template)):
            card = template.iloc[li]

            if card['TYPE'] == 'section':
                if len(card['COMMENT']) > 72 : Warning("Section text exceeds 80 characters, EXTEND will be used.")
                hdr.append(('COMMENT', ('----- ' + card['COMMENT'] + ' ').ljust(72,'-')), end=True)

            elif card['TYPE'] == 'comment':
                hdr.append(('COMMENT', card['VALUE']), end=True)

            elif card['TYPE'] == 'keyword':
                if len(card['VALUE']) + len(card['COMMENT']) > 72:
                    Warning("Section text exceeds 80 characters, EXTEND will be used.")

                if card['DATATYPE'] == 'str':
                    hdr.append((card['KEYWORD'], card['VALUE'], card['COMMENT']), end=True)
                elif card['DATATYPE'] == 'int':
                    hdr.append((card['KEYWORD'], int(card['VALUE']), card['COMMENT']), end=True)
                elif card['DATATYPE'] == 'float':
                    hdr.append((card['KEYWORD'], np.float(card['VALUE']), card['COMMENT']), end=True)

        return hdr

    def to_text(self, file: str) -> None:
        self.totextfile('name.txt', endcard = True, overwrite = True)

    def save(self):
        pass

    def check_empty(self) -> list:
        """
        Return a list of empty required header keywords.

        Returns
        -------
        empty
            List of empty unassigned keywords

        """

        empty = [key for key, value in self.items() if not value or value.isspace()]

        return empty

    @staticmethod
    def verifycsv():
        pass

    @staticmethod
    def gen_template():
        pass

    @classmethod
    def verify(self):
        # Can use .verify('fix')
        # https://docs.astropy.org/en/stable/io/fits/usage/verification.html
        pass

class PUNCHData:
    """PUNCH data object
    Allows for the input of a dictionary of NDCubes for storage and custom methods.
    Used to bundle multiple data sets together to pass through the PUNCH pipeline.

    See Also
    --------
    NDCube : Base container for the PUNCHData object

    Examples
    --------
    >>> from punchbowl.data import PUNCHData
    >>> from ndcube import NDCube

    >>> ndcube_obj = NDCube(data, wcs=wcs, uncertainty=uncertainty, meta=meta, unit=unit)
    >>> data_obj = {'default': ndcube_obj}

    >>> data = PUNCHData(data_obj)
    """

    def __init__(self, data_obj: Union[dict, NDCube, None]):
        """Initialize the PUNCHData object with either an
        empty NDCube object, or a provided NDCube / dictionary
        of NDCube objects

        Parameters
        ----------
        data_obj
            input data object

        Returns
        ----------
        None

        """
        if isinstance(data_obj, dict):
            self._cubes = data_obj
        elif isinstance(data_obj, NDCube):
            self._cubes = {"default": data_obj}
        elif data_obj is None:
            self._cubes = dict()
        else:
            raise Exception("Please specify either an NDCube object, or a dictionary of NDCube objects")

        self._history = History()

    def add_history(self, time: datetime, source: str, comment: str):
        self._history.add_entry(HistoryEntry(time, source, comment))

    @classmethod
    def from_fits(cls, inputs: Union[str, List[str], Dict[str, str]]) -> PUNCHData:
        """
        Populates a PUNCHData object from specified FITS files.
        Specify a filename string, a list of filename strings, or a dictionary of keys and filenames

        Parameters
        ----------
        inputs
            input from which to generate a PUNCHData object
            (filename string, a list of filename strings, or a dictionary of keys and filenames)

        Returns
        -------
        PUNCHData object

        """

        if type(inputs) is str:
            files = {"default": inputs}

        elif type(inputs) is list:
            files = {}
            for file in inputs:
                if type(file) is str:
                    files[file] = file
                else:
                    raise Exception("PUNCHData objects are generated with a list of filename strings.")

        elif type(inputs) is dict:
            files = {}
            for key in inputs:
                if type(inputs[key]) is str:
                    files[key] = inputs[key]
                else:
                    raise Exception("PUNCHData objects are generated with a dictionary of keys and string filenames.")

        else:
            raise Exception("PUNCHData objects are generated with a filename string, a list of filename strings, "
                            "or a dictionary of keys and filenames")

        data_obj = {}

        for key in files:
            with fits.open(files[key]) as hdul:
                data = hdul[0].data
                meta = hdul[0].header
                wcs = WCS(hdul[0].header)
                uncertainty = StdDevUncertainty(hdul[1].data)
                unit = u.ct  # counts
                ndcube_obj = NDCube(data, wcs=wcs, uncertainty=uncertainty, meta=meta, unit=unit)
                data_obj[key] = ndcube_obj

        return cls(data_obj)

    def weight(self, kind: str = "default") -> np.ndarray:
        """
        Generate a corresponding weight map from the uncertainty array

        Parameters
        ----------
        kind
            specified element of the PUNCHData object to generate weights

        Returns
        -------
        weight
            weight map computed from uncertainty array

        """

        return 1./self._cubes[kind].uncertainty.array

    def generate_uncertainties(self, kind: str = "default") -> np.ndarray:
        """
        """
        pass

    def __contains__(self, kind) -> bool:
        return kind in self._cubes

    def __getitem__(self, kind) -> NDCube:
        return self._cubes[kind]

    def __setitem__(self, kind, data) -> None:
        if type(data) is NDCube:
            self._cubes[kind] = data
        else:
            raise Exception("PUNCHData entries must contain NDCube objects.")

    def __delitem__(self, kind) -> None:
        del self._cubes[kind]

    def clear(self) -> None:
        """remove all NDCubes"""
        self._cubes.clear()

    def update(self, other: PUNCHData) -> None:
        """merge two PUNCHData objects"""
        self._cubes.update(other)

    def generate_id(self, kind: str = "default") -> str:
        """
        Dynamically generate an identification string for the given data product,
            using the format 'Ln_ttO_yyyymmddhhmmss'
        Parameters
        ----------
        kind
            specified element of the PUNCHData object to write to file

        Returns
        -------
        id
            output identification string

        """
        observatory = self._cubes[kind].meta['OBSRVTRY']
        file_level = self._cubes[kind].meta['LEVEL']
        type_code = self._cubes[kind].meta['TYPECODE']
        date_obs = self._cubes[kind].date_obs
        date_string = date_obs.strftime("%Y%m%d%H%M%S")

        filename = 'PUNCH_L' + file_level + '_' + type_code + observatory + '_' + date_string

        return filename

    def write(self, filename: str, kind: str = "default", overwrite=True) -> None:
        """
        Write PUNCHData elements to file

        Parameters
        ----------
        filename
            output filename (including path and file extension)
        kind
            specified element of the PUNCHData object to write to file
        overwrite
            True will overwrite an exsiting file, False will create an execption if a file exists

        Returns
        -------
        update_table
            dictionary of pipeline metadata

        """

        if filename.endswith('.fits'):
            self._write_fits(filename, kind, overwrite=overwrite)
        elif filename.endswith('.png'):
            self._write_ql(filename, kind)
        elif filename.endswith('.jpg') or filename.endswith('.jpeg'):
            self._write_ql(filename, kind)
        else:
            raise Exception('Please specify a valid file extension (.fits, .png, .jpg, .jpeg)')

    def _write_fits(self, filename: str, kind: str = "default", overwrite=True) -> None:
        """
        Write PUNCHData elements to FITS files

        Parameters
        ----------
        filename
            output filename (including path and file extension)
        kind
            specified element of the PUNCHData object to write to file
        overwrite
            True will overwrite an exsiting file, False will throw an exeception in that scenario

        Returns
        -------

        """

        # Populate elements to write to file
        data = self._cubes[kind].data
        uncert = self._cubes[kind].uncertainty
        meta = self._cubes[kind].meta
        wcs = self._cubes[kind].wcs

        hdu_data = fits.PrimaryHDU()
        hdu_data.data = data
        # TODO: properly write meta to header
        # for key, value in meta.items():
        #     hdu_data.header[key] = value

        # TODO: remove protected usage by adding a new iterate method
        for entry in self._history._entries:
            hdu_data.header['HISTORY'] = f"{entry.datetime}: {entry.source}, {entry.comment}"

        hdu_uncert = fits.ImageHDU()
        hdu_uncert.data = uncert.array

        hdul = fits.HDUList([hdu_data, hdu_uncert])

        # Write to FITS
        hdul.writeto(filename, overwrite=overwrite)

    def _write_ql(self, filename: str, kind: str = "default") -> None:
        """
        Write an 8-bit scaled version of the specified data array to a PNG file

        Parameters
        ----------
        filename
            output filename (including path and file extension)
        kind
            specified element of the PUNCHData object to write to file

        Returns
        -------
        None

        """

        if self[kind].data.ndim != 2:
            raise Exception("Specified output data should have two-dimensions.")

        # Scale data array to 8-bit values
        output_data = np.int(np.fix(np.interp(self[kind].data, (self[kind].data.min(), self[kind].data.max()), (0, 2**8 - 1))))

        # Write image to file
        matplotlib.image.saveim(filename, output_data)

    def plot(self, kind: str = "default") -> None:
        """
         Generate relevant plots to display or file

         Parameters
         ----------
         kind
             specified element of the PUNCHData object to write to file

         Returns
         -------
         None

         """
        self._cubes[kind].show()

    def get_meta(self, key: str, kind: str = "default") -> Union[str, int, float]:
        """
        Retrieves meta data about a cube
        Parameters
        ----------
        key
        kind

        Returns
        -------

        """
        return self._cubes[kind].meta[key]

    def set_meta(self, key: str, value: Any, kind: str = "default") -> None:
        """
        Retrieves metadata about a cube
        Parameters
        ----------
        key
            specified metadata key
        value
            Updated metadata information
        kind
            specified element of the PUNCHData object to write to file

        Returns
        -------
        None

        """
        self._cubes[kind].meta[key] = value

    def date_obs(self, kind: str = "default") -> datetime:
        return parse_datetime(self._cubes[kind].meta["date-obs"])

