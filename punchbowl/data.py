from __future__ import annotations

import os.path
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

    - TODO - flag option for warnings for unpopulated keywords
    - TODO - more general flag to supress non critical warnings
    - TODO - custom warnings to supress particular types?

    """
    def __init__(self, template=None):
        self._template = template if template else fits.Header()

    @classmethod
    def load(cls, path: str) -> HeaderTemplate:
        """Loads an input template file to generate a header object.

        Parameters
        ----------
        path
            input header template file

        Returns
        -------
        hdr
            astropy.io.fits.Header header object

        """

        if path.endswith('.txt'):
            hdr = HeaderTemplate._load_text(path)
        elif path.endswith('.tsv'):
            hdr = HeaderTemplate._load_tsv(path)
        elif path.endswith('.csv'):
            hdr = HeaderTemplate._load_csv(path)
        else:
            raise Exception('Header template must have a valid file extension. (.txt, .tsv, .csv)'
                            f'Found: {os.path.splitext(path)[1]}')

        template = HeaderTemplate()
        template._template = hdr
        return template

    @staticmethod
    def _load_text(path: str) -> fits.Header:
        """Parses an input human-readable FITS header template and generates an FITS compliant header object.

        Parameters
        ----------
        path
            input header template file

        Returns
        -------
        astropy.io.fits.header
            Header with empty fields
        """
        with open(path, 'r') as f:
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
    def _load_tsv(path: str) -> fits.Header:
        """Parses an input template header tab separated value (TSV) file to generate an astropy header object.

        Parameters
        ----------
        path
            input filename

        Returns
        -------
        astropy.io.fits.header
             header with empty fields
        """

        hdr = fits.Header()

        with open(path, 'r') as csv_file:
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
    def _load_csv(path: str) -> fits.Header:
        """Parses an input template header comma separated value (CSV) file to generate an astropy header object.

        Parameters
        ----------
        path
            input filename

        Returns
        -------
        astropy.io.fits.header
            Header with empty fields
        """
        hdr = fits.Header()

        template = pandas.read_csv(path, keep_default_na=False)

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

    def find_empty(self) -> list:
        """Return a list of empty required header keywords.

        Returns
        -------
        List
            List of unassigned keywords
        """

        empty = [key for key, value in self._template.items() if not value or value.isspace()]

        return empty

    def fill(self, meta_dict: Dict) -> fits.Header:
        out = self._template.copy()
        return out.extend(meta_dict, update=True)


class PUNCHData(NDCube):
    """PUNCH data object
    Allows for the input of a dictionary of NDCubes for storage and custom methods.
    Used to bundle multiple data sets together to pass through the PUNCH pipeline.

    See Also
    --------
    NDCube : Base container for the PUNCHData object
    """

    def __init__(self, data, wcs=None, uncertainty=None, mask=None, meta=None, unit=None, copy=False,
                 history=None, **kwargs):
        """Initialize the PUNCHData object with either an
        empty NDCube object, or a provided NDCube / dictionary
        of NDCube objects

        Parameters
        ----------
        # TODO: fill in

        Returns
        ----------
        None
        """
        super().__init__(data, wcs=wcs, uncertainty=uncertainty, mask=mask, meta=meta, unit=unit, copy=copy, **kwargs)
        self._history = history if history else History()

    def add_history(self, time: datetime, source: str, comment: str):
        self._history.add_entry(HistoryEntry(time, source, comment))

    @classmethod
    def from_fits(cls, path: str) -> PUNCHData:
        """
        Populates a PUNCHData object from specified FITS files.
        Specify a filename string, a list of filename strings, or a dictionary of keys and filenames

        Parameters
        ----------
        path
            filename from which to generate a PUNCHData object

        Returns
        -------
        PUNCHData object
        """

        with fits.open(path) as hdul:
            data = hdul[0].data
            meta = hdul[0].header
            wcs = WCS(hdul[0].header)
            uncertainty = StdDevUncertainty(hdul[1].data)
            unit = u.ct  # counts

        return cls(data, wcs=wcs, uncertainty=uncertainty, meta=meta, unit=unit)

    @property
    def weight(self) -> np.ndarray:
        """
        Generate a corresponding weight map from the uncertainty array

        Returns
        -------
        weight
            weight map computed from uncertainty array
        """

        return 1./self.uncertainty.array

    @property
    def id(self) -> str:
        """Dynamically generate an id string for the given data product, using the format 'Ln_ttO_yyyymmddhhmmss'

        Returns
        -------
        id
            output identification string
        """
        observatory = self.meta['OBSRVTRY']
        file_level = self.meta['LEVEL']
        type_code = self.meta['TYPECODE']
        date_string = self.datetime.strftime("%Y%m%d%H%M%S")
        return 'PUNCH_L' + file_level + '_' + type_code + observatory + '_' + date_string

    def write(self, filename: str, overwrite=True) -> None:
        """
        Write PUNCHData elements to file

        Parameters
        ----------
        filename
            output filename (including path and file extension)
        overwrite
            True will overwrite an exiting file, False will create an exception if a file exists

        Returns
        -------
        update_table
            dictionary of pipeline metadata

        Raises
        -----
        ValueError
            If `filename` does not end in .fits, .png, .jpg, or .jpeg

        """

        if filename.endswith('.fits'):
            self._write_fits(filename, overwrite=overwrite)
        elif filename.endswith('.png'):
            self._write_ql(filename)
        elif filename.endswith('.jpg') or filename.endswith('.jpeg'):
            self._write_ql(filename)
        else:
            raise ValueError('Filename must have a valid file extension (.fits, .png, .jpg, .jpeg). '
                             f'Found: {os.path.splitext(filename)[1]}')

    def _write_fits(self, filename: str, overwrite=True) -> None:
        """
        Write PUNCHData elements to FITS files

        Parameters
        ----------
        filename
            output filename (including path and file extension)
        overwrite
            True will overwrite an exsiting file, False will throw an exeception in that scenario

        Returns
        -------

        """
        hdu_data = fits.PrimaryHDU()
        hdu_data.data = self.data
        # TODO - correct writing meta to header?

        meta = self.create_header("hdr_test_template.txt")

        for key, value in meta.items():
            hdu_data.header[key] = value

        # TODO: remove protected usage by adding a new iterate method
        for entry in self._history._entries:
            hdu_data.header['HISTORY'] = f"{entry.datetime}: {entry.source}, {entry.comment}"

        hdu_uncertainty = fits.ImageHDU()
        hdu_uncertainty.data = self.uncertainty.array

        hdul = fits.HDUList([hdu_data, hdu_uncertainty])

        # Write to FITS
        hdul.writeto(filename, overwrite=overwrite)

    def _write_ql(self, filename: str) -> None:
        """
        Write an 8-bit scaled version of the specified data array to a PNG file

        Parameters
        ----------
        filename
            output filename (including path and file extension)

        Returns
        -------
        None

        """

        if self.data.ndim != 2:
            raise Exception("Specified output data should have two-dimensions.")

        # Scale data array to 8-bit values
        output_data = np.int(np.fix(np.interp(self.data, (self.data.min(), self.data.max()), (0, 2**8 - 1))))

        # Write image to file
        matplotlib.image.saveim(filename, output_data)

    def create_header(self, header_file: str = "") -> fits.Header:
        """
        Validates / generates PUNCHData object metadata using data product header standards

        Parameters
        ----------
        header_file
            specified header template file with which to validate

        """
        # TODO: what does this do when `header_file` is empty?
        return HeaderTemplate(header_file).fill(self.meta)

    @property
    def datetime(self) -> datetime:
        return parse_datetime(self.meta["date-obs"])
