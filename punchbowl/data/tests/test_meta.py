import os
from datetime import datetime

import astropy
import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import GCRS, ICRS, EarthLocation, SkyCoord, get_sun
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.time import Time
from astropy.wcs import WCS
from ndcube import NDCube
from pytest import fixture
from sunpy.coordinates import frames

from punchbowl.data.history import History, HistoryEntry
from punchbowl.data.meta import MetaField, NormalizedMetadata, load_spacecraft_def

TESTDATA_DIR = os.path.dirname(__file__)
SAMPLE_FITS_PATH_UNCOMPRESSED = os.path.join(TESTDATA_DIR, "test_data.fits")
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


def test_normalizedmetadata_get_keys():
    result = NormalizedMetadata.load_template("CB1",
                                              omniheader_path=SAMPLE_OMNIBUS_PATH,
                                              level_spec_path=SAMPLE_LEVEL_PATH,
                                              spacecraft_def_path=SAMPLE_SPACECRAFT_DEF_PATH)
    keys = result.fits_keys
    assert len(keys) == 5


def test_create_level0_normalized_metadata():
    m = NormalizedMetadata.load_template("PM1", "0")
    assert 'DATE-OBS' in m
    assert "OBSRVTRY" in m


def test_create_level3_normalized_metadata():
    m = NormalizedMetadata.load_template("PTM", "3")
    assert "GEOD_LAT" in m
    assert "HGLN_OBS" in m
    assert "HEEX_OBS" in m


def test_normalized_metadata_to_fits_writes_history():
    m = NormalizedMetadata.load_template("PM1", "0")
    m.history.add_now("Test", "does it write?")
    m.history.add_now("Test", "how about twice?")
    m.delete_section("World Coordinate System")
    h = m.to_fits_header()
    assert "HISTORY" in h
    assert "does it write?" in str(h['HISTORY'])
    assert "how about twice" in str(h['HISTORY'])


def test_normalizedmetadata_from_fits_header():
    m = NormalizedMetadata.load_template("PM1", "0")
    m['DESCRPTN'] = 'This is a test!'
    m.history.add_entry(HistoryEntry(datetime(2023, 10, 30, 12, 20), "test", "test comment"))
    m.history.add_entry(HistoryEntry(datetime(2023, 10, 30, 12, 20), "test2", "test comment"))
    m.delete_section("World Coordinate System")
    h = m.to_fits_header()

    recovered = NormalizedMetadata.from_fits_header(h)
    recovered.delete_section("World Coordinate System")

    assert recovered == m


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
