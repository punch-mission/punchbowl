import os
import astropy
from pytest import fixture
from datetime import datetime
from punchbowl.data import PUNCHData, History, HistoryEntry
from ndcube import NDCube
import numpy as np
import pytest


TESTDATA_DIR = os.path.dirname(__file__)
SAMPLE_FITS_PATH = os.path.join(TESTDATA_DIR, "L0_CL1_20211111070246_PUNCHData.fits")
SAMPLE_WRITE_PATH = os.path.join(TESTDATA_DIR, "write_test.fits")


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
    wcs.wcs.ctype = 'WAVE', 'HPLT-TAN', 'HPLN-TAN'
    wcs.wcs.cunit = 'Angstrom', 'deg', 'deg'
    wcs.wcs.cdelt = 0.2, 0.5, 0.4
    wcs.wcs.crpix = 0, 2, 2
    wcs.wcs.crval = 10, 0.5, 1
    wcs.wcs.cname = 'wavelength', 'HPC lat', 'HPC lon'
    nd_obj = NDCube(data=data, wcs=wcs)
    return nd_obj


def test_sample_data_creation(sample_data):
    assert isinstance(sample_data, PUNCHData)


def test_generate_from_filename():
    pd = PUNCHData.from_fits(SAMPLE_FITS_PATH)
    assert isinstance(pd, PUNCHData)


# TODO: fix this test so it runs and passes
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
