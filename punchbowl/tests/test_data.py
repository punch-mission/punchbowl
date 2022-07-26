import os
import astropy
from pytest import fixture
from datetime import datetime
from punchbowl.data import PUNCHData, History, HistoryEntry
from ndcube import NDCube
import numpy as np


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


def test_generate_from_filenamelist():
    fl_list = [SAMPLE_FITS_PATH, SAMPLE_FITS_PATH]
    pd = PUNCHData.from_fits(fl_list)
    assert isinstance(pd, PUNCHData)


def test_generate_from_filenamedict():
    fl_dict = {"default": SAMPLE_FITS_PATH}
    pd = PUNCHData.from_fits(fl_dict)
    assert isinstance(pd, PUNCHData)


def test_generate_from_ndcube(simple_ndcube):
    pd = PUNCHData(simple_ndcube)
    assert isinstance(pd, PUNCHData)


def test_generate_from_ndcubedict(simple_ndcube):
    data_obj = {"default": simple_ndcube}
    pd = PUNCHData(data_obj)
    assert isinstance(pd, PUNCHData)


def test_write_data():
    pd = PUNCHData.from_fits(SAMPLE_FITS_PATH)
    pd.set_meta("LEVEL", 1)
    pd.set_meta("TYPECODE", "XX")
    pd.set_meta("OBSRVTRY", "Y")
    pd.set_meta("VERSION", 0.1)
    pd.set_meta("SOFTVERS", 0.1)
    pd.set_meta("DATE-OBS", str(datetime.now()))
    pd.set_meta("DATE-AQD", str(datetime.now()))
    pd.set_meta("DATE-END", str(datetime.now()))
    pd.set_meta("POL", "M")
    pd.set_meta("STATE", "running")
    pd.set_meta("PROCFLOW", 1)

    pd.write(SAMPLE_WRITE_PATH, kind="default")
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
