import os
from datetime import datetime

import astropy
from astropy.io import fits
from ndcube import NDCube
import numpy as np
import pytest
from pytest import fixture

from punchbowl.data import (
    PUNCHData,
    History,
    HistoryEntry,
    HeaderTemplate,
    HEADER_TEMPLATE_COLUMNS,
)


TESTDATA_DIR = os.path.dirname(__file__)
SAMPLE_FITS_PATH = os.path.join(TESTDATA_DIR, "L0_CL1_20211111070246_PUNCHData.fits")
SAMPLE_WRITE_PATH = os.path.join(TESTDATA_DIR, "write_test.fits")
SAMPLE_HEADER_PATH = os.path.join(TESTDATA_DIR, "hdr_test_template.csv")


@fixture
def sample_data():
    return PUNCHData.from_fits(SAMPLE_FITS_PATH)


@fixture
def simple_ndcube():
    # Taken from NDCube documentation

    # Define data array.
    data = np.random.rand(4, 4, 5)
    # Define WCS transformations in an astropy WCS object.
    wcs = astropy.wcs.WCS(naxis=3)
    wcs.wcs.ctype = "WAVE", "HPLT-TAN", "HPLN-TAN"
    wcs.wcs.cunit = "Angstrom", "deg", "deg"
    wcs.wcs.cdelt = 0.2, 0.5, 0.4
    wcs.wcs.crpix = 0, 2, 2
    wcs.wcs.crval = 10, 0.5, 1
    wcs.wcs.cname = "wavelength", "HPC lat", "HPC lon"
    nd_obj = NDCube(data=data, wcs=wcs)
    return nd_obj


def test_sample_data_creation(sample_data):
    assert isinstance(sample_data, PUNCHData)


def test_generate_from_filename():
    pd = PUNCHData.from_fits(SAMPLE_FITS_PATH)
    assert isinstance(pd, PUNCHData)


def test_write_data(sample_data):
    with pytest.raises(RuntimeWarning):
        sample_data.meta["LEVEL"] = 1
        sample_data.meta["TYPECODE"] = "XX"
        sample_data.meta["OBSRVTRY"] = "Y"
        sample_data.meta["VERSION"] = 0.1
        sample_data.meta["SOFTVERS"] = 0.1
        sample_data.meta["DATE-OBS"] = str(datetime.now())
        sample_data.meta["DATE-AQD"] = str(datetime.now())
        sample_data.meta["DATE-END"] = str(datetime.now())
        sample_data.meta["POL"] = "M"

        sample_data.write(SAMPLE_WRITE_PATH)
        # Check for writing to file? Read back in and compare?


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


@fixture
def empty_header():
    return HeaderTemplate()


@fixture
def simple_header_template():
    return HeaderTemplate.load(SAMPLE_HEADER_PATH)


def test_sample_header_creation(empty_header):
    """An empty PUNCH header object is initialized. Test for no raised errors. Test that the object exists."""
    assert isinstance(empty_header, HeaderTemplate)
    assert np.all(
        empty_header._table.columns.values == HEADER_TEMPLATE_COLUMNS
    ), "doesn't have all the columns"


def test_generate_from_csv_filename():
    """A base PUNCH header object is initialized from a comma separated value file template.
    Test for no raised errors. Test that the object exists."""
    hdr = HeaderTemplate.load(SAMPLE_HEADER_PATH)
    assert isinstance(hdr, HeaderTemplate)


def test_generate_from_invalid_file():
    """A base PUNCH header object is initialized from an invalid input file.
    Test for raised errors. Test that the object does not exist."""
    pass


def test_fill_header(simple_header_template):
    with pytest.warns(RuntimeWarning):
        meta = {"LEVEL": 1}
        header = simple_header_template.fill(meta)
        assert isinstance(header, fits.Header)
        assert header["LEVEL"] == 1


def test_unspecified_header_template():
    with pytest.raises(ValueError):
        h = HeaderTemplate.load("")
        assert isinstance(h, HeaderTemplate)
