import os
import astropy
from pytest import fixture, raises
from datetime import datetime
from punchbowl.data import PUNCHData, HeaderTemplate, History, HistoryEntry
from ndcube import NDCube
import numpy as np
from astropy.io import fits


TESTDATA_DIR = os.path.dirname(__file__)
SAMPLE_FITS_PATH = os.path.join(TESTDATA_DIR, "L0_CL1_20211111070246_PUNCHData.fits")
SAMPLE_TEXT_HEADER_PATH = os.path.join(TESTDATA_DIR, "hdr_test_template.txt")
SAMPLE_TSV_HEADER_PATH = os.path.join(TESTDATA_DIR, "hdr_test_template.tsv")
SAMPLE_CSV_HEADER_PATH = os.path.join(TESTDATA_DIR, "hdr_test_template.csv")
SAMPLE_INVALID_HEADER_PATH = os.path.join(TESTDATA_DIR, "hdr_test_bad.txt")
SAMPLE_WRITE_PATH = os.path.join(TESTDATA_DIR, "write_test.fits")


@fixture
def empty_header():
    return HeaderTemplate(None)

# An empty PUNCH header object is initialized. Test for no raised errors. Test that the object exists.
def test_sample_header_creation(empty_header):
    assert isinstance(empty_header, HeaderTemplate)

# A base PUNCH header object is initialized from a text file template. Test for no raised errors. Test that the object exists.
def test_generate_from_text_filename():
    hdr = HeaderTemplate.load(SAMPLE_TEXT_HEADER_PATH)
    assert isinstance(hdr, fits.Header)

# A base PUNCH header object is initialized from a tab separated value file template. Test for no raised errors. Test that the object exists.
def test_generate_from_tsv_filename():
    hdr = HeaderTemplate.load(SAMPLE_TSV_HEADER_PATH)
    assert isinstance(hdr, fits.Header)

# A base PUNCH header object is initialized from a comma separated value file template. Test for no raised errors. Test that the object exists.
def test_generate_from_csv_filename():
    hdr = HeaderTemplate.load(SAMPLE_CSV_HEADER_PATH)
    assert isinstance(hdr, fits.Header)

# A base PUNCH header object is initialized from an invalid input file. Test for raised errors. Test that the object does not exist.
def test_generate_from_invalid_file():
    with raises(Exception):
        hdr = HeaderTemplate.load(SAMPLE_INVALID_HEADER_PATH)
    # assert not isinstance(hdr, fits.Header)

# A generated PUNCH header object is validated against FITS standards.

# Defining some TBD tests
# Test for...

# No header template file

# >80 char header template file
# Non 2880 char header template file

# Incorrect FITS keywords