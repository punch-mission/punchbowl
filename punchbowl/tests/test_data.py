import os
from datetime import datetime
from collections import OrderedDict

import astropy
import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from pytest import fixture

from ndcube import NDCube

from punchbowl.data import (
    PUNCHData,
    History,
    HistoryEntry,
    NormalizedMetadata,
    MetaField,
    load_spacecraft_def
)

TESTDATA_DIR = os.path.dirname(__file__)
SAMPLE_FITS_PATH_COMPRESSED = os.path.join(TESTDATA_DIR, "test_data.fits")
SAMPLE_WRITE_PATH = os.path.join(TESTDATA_DIR, "write_test.fits")
SAMPLE_OMNIBUS_PATH = os.path.join(TESTDATA_DIR, "omniheader.csv")
SAMPLE_LEVEL_PATH = os.path.join(TESTDATA_DIR, "LevelTest.yaml")
SAMPLE_SPACECRAFT_DEF_PATH = os.path.join(TESTDATA_DIR, "spacecraft.yaml")


def test_metafield_creation_keyword_too_long():
    """ cannot create an invalid metafield"""
    with pytest.raises(ValueError):
        mf = MetaField("TOO LONG KEYWORD",
                       "What's up?", 3, int, False, True, -99)


def test_metafield_creation_kinds_do_not_match():
    """the value, default, and kind must all agree"""
    with pytest.raises(TypeError):
        mf = MetaField("HI",
                       "What's up?", 3, str, False, True, "hi there")

    with pytest.raises(TypeError):
        mf = MetaField("TOO LONG KEYWORD",
                       "What's up?", "hi there", str, False, True, -99)


def test_metafield_update():
    """ you can update a metafield if it is set to be mutable"""
    mf = MetaField("HI", "What's up?", 3, int, False, True, -99)
    assert mf.keyword == "HI"
    assert mf.comment == "What's up?"
    assert mf.value == 3
    assert mf._datatype == int
    assert not mf.nullable
    assert mf._mutable
    assert mf.default == -99

    mf.value = 100
    assert mf.value == 100

    mf.default = -999
    assert mf.default == -999


def test_metafield_not_mutable():
    """ you cannot update immutable metafields"""
    mf = MetaField("HI", "What's up?", 3, int, False, False, -99)
    assert mf.keyword == "HI"
    assert mf.comment == "What's up?"
    assert mf.value == 3
    assert mf._datatype == int
    assert not mf.nullable
    assert not mf._mutable
    assert mf.default == -99

    with pytest.raises(RuntimeError):
        mf.value = 100


def test_metafield_wrong_kind_for_update():
    """ you must abide by the kind for an update"""
    mf = MetaField("HI", "What's up?", 3, int, False, True, -99)
    assert mf.keyword == "HI"
    assert mf.comment == "What's up?"
    assert mf.value == 3
    assert mf._datatype == int
    assert not mf.nullable
    assert mf._mutable
    assert mf.default == -99

    with pytest.raises(TypeError):
        mf.value = "this is invalid"

    with pytest.raises(TypeError):
        mf.default = "this is invalid"


def test_normalizedmetadata_from_template_abq():
    result = NormalizedMetadata.load_template("ABT",
                                              omniheader_path=SAMPLE_OMNIBUS_PATH,
                                              level_spec_path=SAMPLE_LEVEL_PATH,
                                              spacecraft_def_path=SAMPLE_SPACECRAFT_DEF_PATH)
    assert isinstance(result, NormalizedMetadata)
    assert 'Section 1' in result.sections
    assert 'Section 2' in result.sections
    assert 'Section 3' in result.sections
    assert 'Section 4' in result.sections
    assert 'Section 5' not in result.sections

    assert 'KEYOMIT1' not in result
    assert 'KEYOMIT2' not in result
    assert result['KEYALTER'].value == 2
    assert result['OTHERKEY'].value == 'Test'
    assert result['TITLE'].value == 'My name is test-0'


def test_normalizedmetadata_from_template_cb1():
    result = NormalizedMetadata.load_template("CB1",
                                              omniheader_path=SAMPLE_OMNIBUS_PATH,
                                              level_spec_path=SAMPLE_LEVEL_PATH,
                                              spacecraft_def_path=SAMPLE_SPACECRAFT_DEF_PATH)
    assert isinstance(result, NormalizedMetadata)
    assert 'Section 1' in result.sections
    assert 'Section 2' in result.sections
    assert 'Section 3' in result.sections
    assert 'Section 4' in result.sections
    assert 'Section 5' not in result.sections

    assert 'KEYOMIT1' not in result
    assert 'KEYOMIT2' not in result
    assert result['KEYALTER'].value == 3
    assert result['OTHERKEY'].value == 'No test'
    assert result['TITLE'].value == 'A default value'
    assert result['TITLE'].comment == 'with a default comment'


def test_normalizedmetadata_to_fits_header(tmpdir):
    result = NormalizedMetadata.load_template("CB1",
                                              omniheader_path=SAMPLE_OMNIBUS_PATH,
                                              level_spec_path=SAMPLE_LEVEL_PATH,
                                              spacecraft_def_path=SAMPLE_SPACECRAFT_DEF_PATH)
    header = result.to_fits_header()
    header.tofile(os.path.join(str(tmpdir), "test.fits"), overwrite=True)
    assert isinstance(header, fits.Header)


@fixture
def sample_wcs() -> WCS:
    """
    Generate a sample WCS for testing
    """

    def _sample_wcs(naxis=2, crpix=(0, 0), crval=(0, 0), cdelt=(1, 1),
                    ctype=("HPLN-ARC", "HPLT-ARC")):
        generated_wcs = WCS(naxis=naxis)

        generated_wcs.wcs.crpix = crpix
        generated_wcs.wcs.crval = crval
        generated_wcs.wcs.cdelt = cdelt
        generated_wcs.wcs.ctype = ctype

        return generated_wcs

    return _sample_wcs


@fixture
def sample_data():
    return PUNCHData.from_fits(SAMPLE_FITS_PATH_COMPRESSED)


@fixture
def sample_data_random(shape: tuple = (50, 50)) -> np.ndarray:
    """
    Generate some random data for testing
    """
    return np.random.random(shape)


@fixture
def simple_ndcube():
    # Taken from NDCube documentation
    data = np.random.rand(4, 4, 5)
    wcs = astropy.wcs.WCS(naxis=3)
    wcs.wcs.ctype = "WAVE", "HPLT-TAN", "HPLN-TAN"
    wcs.wcs.cunit = "Angstrom", "deg", "deg"
    wcs.wcs.cdelt = 0.2, 0.5, 0.4
    wcs.wcs.crpix = 0, 2, 2
    wcs.wcs.crval = 10, 0.5, 1
    wcs.wcs.cname = "wavelength", "HPC lat", "HPC lon"
    nd_obj = NDCube(data=data, wcs=wcs)
    return nd_obj


@fixture
def sample_punchdata():
    """
    Generate a sample PUNCHData object for testing
    """

    def _sample_punchdata(shape=(50, 50), level=0):
        data = np.random.random(shape).astype(np.float32)
        uncertainty = StdDevUncertainty(np.sqrt(np.abs(data)))
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"
        wcs.wcs.cunit = "deg", "deg"
        wcs.wcs.cdelt = 0.1, 0.1
        wcs.wcs.crpix = 0, 0
        wcs.wcs.crval = 1, 1
        wcs.wcs.cname = "HPC lon", "HPC lat"

        # meta = NormalizedMetadata({"LEVEL": str(level),
        #                            'OBSRVTRY': 'Y',
        #                            'TYPECODE': 'XX',
        #                            'DATE-OBS': })
        meta = NormalizedMetadata.load_template("PM1", "0")
        meta['DATE-OBS'] = str(datetime(2023, 1, 1, 0, 0, 1))
        return PUNCHData(data=data, uncertainty=uncertainty, wcs=wcs, meta=meta)

    return _sample_punchdata


@fixture
def sample_punchdata_list(sample_punchdata):
    """
    Generate a list of sample PUNCHData objects for testing
    """
    sample_pd1 = sample_punchdata()
    sample_pd2 = sample_punchdata()
    return [sample_pd1, sample_pd2]


def test_sample_data_creation(sample_data):
    assert isinstance(sample_data, PUNCHData)


def test_generate_from_filename_compressed():
    pd = PUNCHData.from_fits(SAMPLE_FITS_PATH_COMPRESSED)
    assert isinstance(pd, PUNCHData)
    assert isinstance(pd.data, np.ndarray)


def test_write_data(sample_punchdata):
    sample_data = sample_punchdata()
    sample_data.meta["LEVEL"] = "1"
    sample_data.meta["TYPECODE"] = "CL"
    sample_data.meta["OBSRVTRY"] = "1"
    sample_data.meta["PIPEVRSN"] = "0.1"
    sample_data.meta["DATE-OBS"] = str(datetime.now())
    sample_data.meta["DATE-END"] = str(datetime.now())

    sample_data.write(SAMPLE_WRITE_PATH)
    assert os.path.isfile(SAMPLE_WRITE_PATH)


def test_generate_data_statistics(sample_punchdata):
    sample_data = sample_punchdata()

    sample_data._update_statistics()

    assert sample_data.meta['DATAZER'].value == len(np.where(sample_data.data == 0)[0])


def test_read_write_uncertainty_data(sample_punchdata):
    sample_data = sample_punchdata()
    uncertainty = StdDevUncertainty(np.sqrt(np.abs(sample_data.data)).astype('uint8'))
    sample_data.uncertainty = uncertainty

    sample_data.write(SAMPLE_WRITE_PATH)

    pdata_read_data = PUNCHData.from_fits(SAMPLE_WRITE_PATH)

    with fits.open(SAMPLE_WRITE_PATH) as hdul:
        fitsio_read_uncertainty = hdul[2].data
        fitsio_read_header = hdul[2].header

    assert fitsio_read_header['BITPIX'] == 8

    assert sample_data.uncertainty.array.dtype == 'uint8'
    assert pdata_read_data.uncertainty.array.dtype == 'uint8'
    assert fitsio_read_uncertainty.dtype == 'uint8'


def test_filename_base_generation(sample_punchdata):
    actual = sample_punchdata().filename_base
    expected = "PUNCH_L0_PM1_20230101000001"
    assert actual == expected



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


def test_load_spacecraft_yaml():
    """ tests that spacecraft yaml can be loaded and is well-formed"""
    sc_yaml = load_spacecraft_def(path=SAMPLE_SPACECRAFT_DEF_PATH)
    for i in range(1, 5):
        assert str(i) in sc_yaml
    assert "M" in sc_yaml
    assert "N" in sc_yaml
    assert "T" in sc_yaml

    for _, v in sc_yaml.items():
        assert "craftname" in v
        assert "crafttype" in v
        assert "obsname" in v


def test_create_level0_normalized_metadata():
    m = NormalizedMetadata.load_template("PM1", "0")
    assert 'DATE-OBS' in m
    assert "OBSRVTRY" in m


def test_normalized_metadata_to_fits_writes_history():
    m = NormalizedMetadata.load_template("PM1", "0")
    m.history.add_now("Test", "does it write?")
    m.history.add_now("Test", "how about twice?")
    h = m.to_fits_header()
    assert "HISTORY" in h
    assert "does it write?" in str(h['HISTORY'])
    assert "how about twice" in str(h['HISTORY'])


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


def test_from_fits_for_metadata(tmpdir):
    m = NormalizedMetadata.load_template("PM1", "0")
    m.history.add_now("Test", "does it write?")
    m.history.add_now("Test", "how about twice?")
    m['DESCRPTN'] = 'This is a test!'
    m['CHECKSUM'] = ''
    m['DATASUM'] = ''
    h = m.to_fits_header()

    path = os.path.join(tmpdir, "from_fits_test.fits")
    d = PUNCHData(np.zeros((2048, 2048), dtype=np.int16), WCS(h), m)
    d.write(path)

    loaded = PUNCHData.from_fits(path)
    loaded.meta['LATPOLE'] = 0.0  # a hackish way to circumvent the LATPOLE being moved by astropy
    assert loaded.meta == m


def test_normalizedmetadata_from_fits_header():
    m = NormalizedMetadata.load_template("PM1", "0")
    m['DESCRPTN'] = 'This is a test!'
    m.history.add_entry(HistoryEntry(datetime(2023, 10, 30, 12, 20), "test", "test comment"))
    m.history.add_entry(HistoryEntry(datetime(2023, 10, 30, 12, 20), "test2", "test comment"))
    h = m.to_fits_header()

    recovered = NormalizedMetadata.from_fits_header(h)

    assert recovered == m


def test_empty_history_from_fits_header():
    m = NormalizedMetadata.load_template("PM1", "0")
    h = m.to_fits_header()

    assert History.from_fits_header(h) == History()


def test_filled_history_from_fits_header(tmpdir):
    constructed_history = History()
    constructed_history.add_entry(HistoryEntry(datetime(2023, 10, 30, 12, 20), "test", "test comment"))
    constructed_history.add_entry(HistoryEntry(datetime(2023, 10, 30, 12, 20), "test2", "test comment"))

    m = NormalizedMetadata.load_template("PM1", "0")
    m.history.add_entry(HistoryEntry(datetime(2023, 10, 30, 12, 20), "test", "test comment"))
    m.history.add_entry(HistoryEntry(datetime(2023, 10, 30, 12, 20), "test2", "test comment"))
    h = m.to_fits_header()

    assert History.from_fits_header(h) == constructed_history
