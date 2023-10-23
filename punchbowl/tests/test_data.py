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
SAMPLE_FITS_PATH_UNCOMPRESSED = os.path.join(TESTDATA_DIR, "PUNCH_L2_WQM_20080103071000_uncomp.fits")
SAMPLE_FITS_PATH_COMPRESSED = os.path.join(TESTDATA_DIR, "PUNCH_L2_WQM_20080103071000.fits")
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


# def test_metasection_creation():
#     fields = OrderedDict()
#     fields['key1'] = MetaField("key1", "comment", 4, int, False, False, 3)
#     fields['key2'] = MetaField("key2", "comment", 4, int, False, False, 3)
#     ms = MetaSection("section name", fields)
#     assert 'key1' in ms
#     assert 'key2' in ms
#     assert

# # Test fixtures
# @fixture
# def sample_wcs() -> WCS:
#     """
#     Generate a sample WCS for testing
#     """
#
#     def _sample_wcs(naxis=2, crpix=(0, 0), crval=(0, 0), cdelt=(1, 1),
#                     ctype=("HPLN-ARC", "HPLT-ARC")):
#         generated_wcs = WCS(naxis=naxis)
#
#         generated_wcs.wcs.crpix = crpix
#         generated_wcs.wcs.crval = crval
#         generated_wcs.wcs.cdelt = cdelt
#         generated_wcs.wcs.ctype = ctype
#
#         return generated_wcs
#
#     return _sample_wcs
#
#
# @fixture
# def sample_data():
#     return PUNCHData.from_fits(SAMPLE_FITS_PATH_COMPRESSED)
#
#
# @fixture
# def sample_data_random(shape: tuple = (50, 50)) -> np.ndarray:
#     """
#     Generate some random data for testing
#     """
#     return np.random.random(shape)
#
#
# @fixture
# def simple_ndcube():
#     # Taken from NDCube documentation
#     data = np.random.rand(4, 4, 5)
#     wcs = astropy.wcs.WCS(naxis=3)
#     wcs.wcs.ctype = "WAVE", "HPLT-TAN", "HPLN-TAN"
#     wcs.wcs.cunit = "Angstrom", "deg", "deg"
#     wcs.wcs.cdelt = 0.2, 0.5, 0.4
#     wcs.wcs.crpix = 0, 2, 2
#     wcs.wcs.crval = 10, 0.5, 1
#     wcs.wcs.cname = "wavelength", "HPC lat", "HPC lon"
#     nd_obj = NDCube(data=data, wcs=wcs)
#     return nd_obj
#
#
# @fixture
# def sample_punchdata():
#     """
#     Generate a sample PUNCHData object for testing
#     """
#
#     def _sample_punchdata(shape=(50, 50), level=0):
#         data = np.random.random(shape)
#         uncertainty = StdDevUncertainty(np.sqrt(np.abs(data)))
#         wcs = WCS(naxis=2)
#         wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"
#         wcs.wcs.cunit = "deg", "deg"
#         wcs.wcs.cdelt = 0.1, 0.1
#         wcs.wcs.crpix = 0, 0
#         wcs.wcs.crval = 1, 1
#         wcs.wcs.cname = "HPC lon", "HPC lat"
#
#         meta = NormalizedMetadata({"LEVEL": str(level),
#                                    'OBSRVTRY': 'Y',
#                                    'TYPECODE': 'XX',
#                                    'DATE-OBS': str(datetime(2023, 1, 1, 0, 0, 1))},
#                                   required_fields=PUNCH_REQUIRED_META_FIELDS)
#         return PUNCHData(data=data, uncertainty=uncertainty, wcs=wcs, meta=meta)
#
#     return _sample_punchdata
#
#
# @fixture
# def sample_punchdata_list(sample_punchdata):
#     """
#     Generate a list of sample PUNCHData objects for testing
#     """
#
#     sample_pd1 = sample_punchdata()
#     sample_pd2 = sample_punchdata()
#     return [sample_pd1, sample_pd2]
#
#
# def test_sample_data_creation(sample_data):
#     assert isinstance(sample_data, PUNCHData)
#
#
# def test_generate_from_filename_uncompressed():
#     pd = PUNCHData.from_fits(SAMPLE_FITS_PATH_UNCOMPRESSED)
#     assert isinstance(pd, PUNCHData)
#     assert isinstance(pd.data, np.ndarray)
#
#
# def test_generate_from_filename_compressed():
#     pd = PUNCHData.from_fits(SAMPLE_FITS_PATH_COMPRESSED)
#     assert isinstance(pd, PUNCHData)
#     assert isinstance(pd.data, np.ndarray)
#
#
# def test_write_data(sample_data):
#     with pytest.warns(RuntimeWarning):
#         sample_data.meta["LEVEL"] = 1
#         sample_data.meta["TYPECODE"] = "CL"
#         sample_data.meta["OBSRVTRY"] = "1"
#         sample_data.meta["VERSION"] = 0.1
#         sample_data.meta["SOFTVERS"] = 0.1
#         sample_data.meta["DATE-OBS"] = str(datetime.now())
#         sample_data.meta["DATE-AQD"] = str(datetime.now())
#         sample_data.meta["DATE-END"] = str(datetime.now())
#         sample_data.meta["POL"] = "M"
#
#         sample_data.write(SAMPLE_WRITE_PATH)
#         assert os.path.isfile(SAMPLE_WRITE_PATH)
#
#
# def test_filename_base_generation(sample_punchdata):
#     actual = sample_punchdata().filename_base
#     expected = "PUNCH_L0_XXY_20230101000001"
#     assert actual == expected
#
#

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

#
# def test_generate_from_filename():
#     """A base PUNCH header object is initialized from a comma separated value file template.
#     Test for no raised errors. Test that the object exists."""
#     hdr = HeaderTemplate.from_file(SAMPLE_OMNIBUS_PATH, SAMPLE_LEVEL1_PATH, "CL1")
#     assert isinstance(hdr, HeaderTemplate)
#     assert isinstance(hdr.omniheader, pd.DataFrame)
#     assert isinstance(hdr.level_definition, dict)
#     assert isinstance(hdr.product_definition, dict)
#
#
# def test_fill_header():
#     hdr = HeaderTemplate.from_file(SAMPLE_OMNIBUS_PATH, SAMPLE_LEVEL1_PATH, "CL1")
#     meta = NormalizedMetadata({"LEVEL": 1})
#     header = hdr.fill(meta)
#     assert isinstance(header, fits.Header)
#     assert header["LEVEL"] == 1
#     assert header['TITLE'] == 'Test'
#     assert header['NAXIS'] == 3
#     assert 'GEOD_LAT' not in header
#
#
# def test_header_selection_based_on_level():
#     """Tests if the header can be selected automatically based on the product level"""
#     # Modified from NDCube documentation
#     data = np.random.rand(4, 4, 5)
#     wcs = astropy.wcs.WCS(naxis=3)
#     wcs.wcs.ctype = "WAVE", "HPLT-TAN", "HPLN-TAN"
#     wcs.wcs.cunit = "Angstrom", "deg", "deg"
#     wcs.wcs.cdelt = 0.2, 0.5, 0.4
#     wcs.wcs.crpix = 0, 2, 2
#     wcs.wcs.crval = 10, 0.5, 1
#     wcs.wcs.cname = "wavelength", "HPC lat", "HPC lon"
#
#     meta = NormalizedMetadata({"LEVEL": 1,
#                                'DATE-OBS': str(datetime.now()),
#                                "OBSRVTRY": 1,
#                                "TYPECODE": "CL"}, required_fields=PUNCH_REQUIRED_META_FIELDS)
#     data = PUNCHData(data=data, wcs=wcs, meta=meta)
#
#     header = data.create_header(SAMPLE_OMNIBUS_PATH, SAMPLE_LEVEL1_PATH)
#     assert header['LEVEL'] == 1
#
#
# def test_normalizedmetadata_access_not_case_sensitive():
#     contents = {"hi": "there", "NaME": "marcus", "AGE": 27}
#     example = NormalizedMetadata(contents)
#
#     assert example["hi"] == "there"
#     assert example["Hi"] == "there"
#     assert example["hI"] == "there"
#     assert example["HI"] == "there"
#
#     assert example["age"] == 27
#
#     assert example['name'] == 'marcus'
#
#
# def test_normalizedmetadata_add_new_key():
#     empty = NormalizedMetadata(dict())
#     assert len(empty) == 0
#     assert "name" not in empty
#     empty['NAmE'] = "marcus"
#     assert "nAMe" in empty
#     assert len(empty) == 1
#
#
# def test_normalizedmetadata_delete_key():
#     example = NormalizedMetadata({"key": "value"})
#     assert "key" in example
#     assert len(example) == 1
#     del example['key']
#     assert "key" not in example
#     assert len(example) == 0
#
#
# def test_normalizedmetadata_missing_required_fields_fails():
#     required_fields = {"age", "name"}
#     with pytest.raises(RuntimeError):
#         NormalizedMetadata({"key": "value"}, required_fields=required_fields)
#
#
# def test_normalizedmetadata_has_all_required_fields():
#     required_fields = {"age", "name"}
#     example = NormalizedMetadata({'name': 'Marcus', 'age': 27}, required_fields=required_fields)
#     for key in required_fields:
#         assert key in example
