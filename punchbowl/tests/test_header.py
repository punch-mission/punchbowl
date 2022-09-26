import os
import astropy
import pytest
from pytest import fixture, raises
from datetime import datetime
from punchbowl.data import PUNCHData, HeaderTemplate, History, HistoryEntry, HEADER_TEMPLATE_COLUMNS
from ndcube import NDCube
import numpy as np
from astropy.io import fits


TESTDATA_DIR = os.path.dirname(__file__)
SAMPLE_FITS_PATH = os.path.join(TESTDATA_DIR, "L0_CL1_20211111070246_PUNCHData.fits")
SAMPLE_HEADER_PATH = os.path.join(TESTDATA_DIR, "hdr_test_template.csv")
SAMPLE_WRITE_PATH = os.path.join(TESTDATA_DIR, "write_test.fits")


@fixture
def empty_header():
    return HeaderTemplate()


@fixture
def simple_header_template():
    return HeaderTemplate.load(SAMPLE_HEADER_PATH)


def test_sample_header_creation(empty_header):
    """An empty PUNCH header object is initialized. Test for no raised errors. Test that the object exists."""
    assert isinstance(empty_header, HeaderTemplate)
    assert np.all(empty_header._table.columns.values == HEADER_TEMPLATE_COLUMNS), "doesn't have all the columns"


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
    with pytest.raises(RuntimeWarning):
        meta = {"LEVEL": 1}
        header = simple_header_template.fill(meta)
        assert isinstance(header, fits.Header)
        assert header['LEVEL'] == 1

# TODO: remove TBD tests
# Defining some TBD tests

# A generated PUNCH header object is validated against FITS standards, and corrected.
#def test_verify_header():
#    hdr = HeaderTemplate.load(SAMPLE_TEXT_HEADER_PATH)
#    hdr.verify()
#    assert isinstance(hdr, fits.Header)

# Test for...

# No header template file

# >80 char header template file
# Non 2880 char header template file

# Incorrect FITS keywords