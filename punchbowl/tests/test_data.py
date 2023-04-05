import os
from datetime import datetime

import astropy
import numpy as np
import pytest
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from pytest import fixture

from punchbowl.data import (
    PUNCHData,
    History,
    HistoryEntry,
    HeaderTemplate,
    HEADER_TEMPLATE_COLUMNS,
    NormalizedMetadata,
    PUNCH_REQUIRED_META_FIELDS
)

TESTDATA_DIR = os.path.dirname(__file__)
SAMPLE_HEADER_PATH = os.path.join(TESTDATA_DIR, "hdr_test_template.csv")

# Test fixtures
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
def sample_data_random(shape: tuple = (50, 50)) -> np.ndarray:
    """
    Generate some random data for testing
    """
    return np.random.random(shape)


@fixture
def sample_punchdata():
    """
    Generate a sample PUNCHData object for testing
    """

    def _sample_punchdata(shape=(50, 50), level=0):
        data = np.random.random(shape)
        uncertainty = StdDevUncertainty(np.sqrt(np.abs(data)))
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"
        wcs.wcs.cunit = "deg", "deg"
        wcs.wcs.cdelt = 0.1, 0.1
        wcs.wcs.crpix = 0, 0
        wcs.wcs.crval = 1, 1
        wcs.wcs.cname = "HPC lon", "HPC lat"

        meta = NormalizedMetadata({"LEVEL": str(level),
                                   'OBSRVTRY': 'Y',
                                   'TYPECODE': 'XX',
                                   'DATE-OBS': str(datetime(2023, 1, 1, 0, 0, 1))},
                                  required_fields=PUNCH_REQUIRED_META_FIELDS)
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


def test_filename_base_generation(sample_punchdata):
    actual = sample_punchdata().filename_base
    expected = "PUNCH_L0_XXY_20230101000001"
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


def test_fill_header(simple_header_template):
    with pytest.warns(RuntimeWarning):
        meta = NormalizedMetadata({"LEVEL": 1})
        header = simple_header_template.fill(meta)
        assert isinstance(header, fits.Header)
        assert header["LEVEL"] == 1


def test_unspecified_header_template():
    with pytest.raises(ValueError):
        h = HeaderTemplate.load("")
        assert isinstance(h, HeaderTemplate)


@pytest.mark.parametrize("level", [0, 1, 2])
def test_header_selection_based_on_level(level: int):
    """Tests if the header can be selected automatically based on the product level"""
    # Modified from NDCube documentation
    data = np.random.rand(4, 4, 5)
    wcs = astropy.wcs.WCS(naxis=3)
    wcs.wcs.ctype = "WAVE", "HPLT-TAN", "HPLN-TAN"
    wcs.wcs.cunit = "Angstrom", "deg", "deg"
    wcs.wcs.cdelt = 0.2, 0.5, 0.4
    wcs.wcs.crpix = 0, 2, 2
    wcs.wcs.crval = 10, 0.5, 1
    wcs.wcs.cname = "wavelength", "HPC lat", "HPC lon"

    meta = NormalizedMetadata({"LEVEL": level,
                               'DATE-OBS': str(datetime.now()),
                               "OBSRVTRY": "X",
                               "TYPECODE": "YY"}, required_fields=PUNCH_REQUIRED_META_FIELDS)
    data = PUNCHData(data=data, wcs=wcs, meta=meta)

    header = data.create_header(None)
    assert header['LEVEL'] == level


def test_normalizedmetadata_access_not_case_sensitive():
    contents = {"hi": "there", "NaME": "marcus", "AGE": 27}
    example = NormalizedMetadata(contents)

    assert example["hi"] == "there"
    assert example["Hi"] == "there"
    assert example["hI"] == "there"
    assert example["HI"] == "there"

    assert example["age"] == 27

    assert example['name'] == 'marcus'


def test_normalizedmetadata_add_new_key():
    empty = NormalizedMetadata(dict())
    assert len(empty) == 0
    assert "name" not in empty
    empty['NAmE'] = "marcus"
    assert "nAMe" in empty
    assert len(empty) == 1


def test_normalizedmetadata_delete_key():
    example = NormalizedMetadata({"key": "value"})
    assert "key" in example
    assert len(example) == 1
    del example['key']
    assert "key" not in example
    assert len(example) == 0


def test_normalizedmetdata_missing_required_fields_fails():
    required_fields = {"age", "name"}
    with pytest.raises(RuntimeError):
        NormalizedMetadata({"key": "value"}, required_fields=required_fields)


def test_normalizedemetadata_has_all_required_fields():
    required_fields = {"age", "name"}
    example = NormalizedMetadata({'name': 'Marcus', 'age': 27}, required_fields=required_fields)
    for key in required_fields:
        assert key in example
