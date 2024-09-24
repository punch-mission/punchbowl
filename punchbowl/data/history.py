from __future__ import annotations

from datetime import datetime
from collections import namedtuple

from astropy.io.fits import Header
from dateutil.parser import parse as parse_datetime

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
        return "\n".join([f"{e.datetime:%Y-%m-%dT%H:%M:%S} => {e.source} => {e.comment}|" for e in self._entries])

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
            entries = "".join([h for h in head["HISTORY"][1:]]).split("|")[:-1]  # noqa: C416
            for entry in entries:
                dt, source, comment = entry.split(" => ")
                dt = parse_datetime(dt)
                out.add_entry(HistoryEntry(dt, source, comment))
        return out
