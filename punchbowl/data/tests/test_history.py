from datetime import datetime

import pytest
from astropy.wcs import WCS
from pytest import fixture

from punchbowl.data.history import History, HistoryEntry
from punchbowl.data.meta import NormalizedMetadata


@fixture
def empty_history():
    return History()


def test_history_add_one(empty_history):
    entry = HistoryEntry(datetime.now(), "test", "dummy")
    assert len(empty_history) == 0
    empty_history.add_entry(entry)
    assert len(empty_history) == 1
    assert empty_history.most_recent().source == "test"
    assert empty_history.most_recent().comment == "dummy"
    assert empty_history.most_recent() == empty_history[-1]
    empty_history.clear()
    assert len(empty_history) == 0


def test_history_iterate(empty_history):
    empty_history.add_entry(HistoryEntry(datetime.now(), "0", "0"))
    empty_history.add_entry(HistoryEntry(datetime.now(), "1", "1"))
    empty_history.add_entry(HistoryEntry(datetime.now(), "2", "2"))

    for i, entry in enumerate(empty_history):
        assert entry.comment == str(i), "history objects not read in order"


def test_empty_history_equal():
    assert History() == History()


def test_history_equality():
    entry = HistoryEntry(datetime(2023, 10, 30, 12, 20), "test", "test comment")

    h1 = History()
    h1.add_entry(entry)

    h2 = History()
    h2.add_entry(entry)

    assert h1 == h2


def test_history_not_equals_if_different():
    h1 = History()
    h1.add_now("Test", "one")

    h2 = History()
    h2.add_now("Test", "two")

    assert h1 != h2


def test_history_cannot_compare_to_nonhistory_type():
    h1 = History()
    h2 = {"Not": "History"}
    with pytest.raises(TypeError):
        h1 == h2


def test_empty_history_from_fits_header():
    m = NormalizedMetadata.load_template("PM1", "0")
    m['DATE-OBS'] = "2024-01-01T00:00:00"
    m.delete_section("World Coordinate System")
    h = m.to_fits_header(wcs=None)

    assert History.from_fits_header(h) == History()


def test_filled_history_from_fits_header(tmpdir):
    constructed_history = History()
    constructed_history.add_entry(HistoryEntry(datetime(2023, 10, 30, 12, 20), "test", "test comment"))
    constructed_history.add_entry(HistoryEntry(datetime(2023, 10, 30, 12, 20), "test2", "test comment"))

    m = NormalizedMetadata.load_template("PM1", "0")
    m.history.add_entry(HistoryEntry(datetime(2023, 10, 30, 12, 20), "test", "test comment"))
    m.history.add_entry(HistoryEntry(datetime(2023, 10, 30, 12, 20), "test2", "test comment"))
    m.delete_section("World Coordinate System")
    h = m.to_fits_header()

    assert History.from_fits_header(h) == constructed_history
