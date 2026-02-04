import os
from datetime import UTC, datetime
from collections import Counter

import numpy as np
import pytest
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS, DistortionLookupTable
from astropy.coordinates import get_body
from astropy.time import Time

from punchbowl.data.history import HistoryEntry
from punchbowl.data.meta import (
    MetaField,
    NormalizedMetadata,
    check_moon_in_fov,
    construct_all_product_codes,
    load_spacecraft_def,
)

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
        MetaField("TOO LONG KEYWORD",
                  "What's up?", 3, int, False, True, -99)


def test_metafield_creation_kinds_do_not_match():
    """the value, default, and kind must all agree"""
    with pytest.raises(TypeError):
        MetaField("HI",
                  "What's up?", 3, str, False, True, "hi there")

    with pytest.raises(TypeError):
        MetaField("TOO LONG KEYWORD",
                  "What's up?", "hi there", str, False, True, -99)


def test_metafield_update():
    """ you can update a metafield if it is set to be mutable"""
    mf = MetaField("HI", "What's up?", 3, int, False, True, -99)
    assert mf.keyword == "HI"
    assert mf.comment == "What's up?"
    assert mf.value == 3
    assert mf._datatype is int
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
    assert mf._datatype is int
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
    assert mf._datatype is int
    assert not mf.nullable
    assert mf._mutable
    assert mf.default == -99

    with pytest.raises(TypeError):
        mf.value = "this is invalid"

    with pytest.raises(TypeError):
        mf.default = "this is invalid"

def test_metafield_hashes():
    mf1 = MetaField("HI", "What's up?", 3, int, False, True, -99)
    mf2 = MetaField("BYE", "What's up?", 3, int, False, True, -99)
    mf3 = MetaField("HI", "What's up?", 3, int, False, True, -99)

    assert hash(mf1) != hash(mf2)
    assert mf1 != mf2
    assert mf1 == mf3
    assert hash(mf1) == hash(mf3)


def test_normalizedmetadata_hashes():
    result = NormalizedMetadata.load_template("CB1",
                                              omniheader_path=SAMPLE_OMNIBUS_PATH,
                                              level_spec_path=SAMPLE_LEVEL_PATH,
                                              spacecraft_def_path=SAMPLE_SPACECRAFT_DEF_PATH)
    assert isinstance(hash(result), int)

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
    assert 'Section 6' not in result.sections

    assert 'KEYOMIT1' not in result
    assert 'KEYOMIT2' not in result
    assert result['KEYALTER'].value == 2
    assert int(result['KEYALTER']) == 2
    assert result['OTHERKEY'].value == 'Test'
    with pytest.raises(TypeError):
        int(result['OTHERKEY'])
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

    assert recovered['HISTORY'] == m['HISTORY']


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


def test_construct_all_product_codes():
    codes = construct_all_product_codes(level="0")
    assert isinstance(codes, list)
    assert len(codes) == 36


def test_fits_header_compliance():
    code="PM1"
    level="0"

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.1, 0.1
    wcs.wcs.crpix = 0, 0
    wcs.wcs.crval = 1, 1
    wcs.wcs.cname = "HPC lon", "HPC lat"

    m = NormalizedMetadata.load_template(code, level)

    m["DATE-OBS"] = str(datetime.now(UTC))
    m["DATE-END"] = str(datetime.now(UTC))

    h = m.to_fits_header(wcs = wcs)

    assert h[0] == 'T'

    allowed_duplicates = {'COMMENT', 'HISTORY'}
    keyword_counts = Counter(key for key in h.keys() if key not in allowed_duplicates)
    keyword_duplicates = [keyword for keyword, count in keyword_counts.items() if count > 1]
    assert not keyword_duplicates

def test_fits_header_has_both_distortion_models():
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.1, 0.1
    wcs.wcs.crpix = 0, 0
    wcs.wcs.crval = 1, 1
    wcs.wcs.cname = "HPC lon", "HPC lat"

    num_bins = 100
    xbins = np.random.random((num_bins, num_bins))
    ybins = np.random.random((num_bins, num_bins))

    r = np.linspace(0, 2048, num_bins + 1)
    c = np.linspace(0, 2048, num_bins + 1)
    r = (r[1:] + r[:-1]) / 2
    c = (c[1:] + c[:-1]) / 2
    err_px, err_py = r, c

    cpdis1 = DistortionLookupTable(
            -xbins.astype(np.float32), (0, 0), (err_px[0], err_py[0]),
            ((err_px[1] - err_px[0]), (err_py[1] - err_py[0])),
        )
    cpdis2 = DistortionLookupTable(
        -ybins.astype(np.float32), (0, 0), (err_px[0], err_py[0]),
        ((err_px[1] - err_px[0]), (err_py[1] - err_py[0])),
    )

    wcs.cpdis1 = cpdis1
    wcs.cpdis2 = cpdis2

    code, level = "PM1", "1"
    m = NormalizedMetadata.load_template(code, level)

    m["DATE-OBS"] = str(datetime.now(UTC))
    m["DATE-END"] = str(datetime.now(UTC))

    h = m.to_fits_header(wcs = wcs)

    assert "CPDIS1" in h
    assert "CPDIS2" in h
    assert "CPDIS1A" in h
    assert "CPDIS2A" in h

    assert "DP1" in h
    assert "DP2" in h
    assert "DP1A" in h
    assert "DP2A" in h


def test_check_moon_in_fov():
    with fits.open(SAMPLE_FITS_PATH_UNCOMPRESSED) as hdul:
        data = hdul[1].data
        header = hdul[1].header
    times, angles, fov_half, in_fov, ang_c, xpix, ypix = check_moon_in_fov(
        time_obs_start="2025-04-10 00:00:00",
        time_end="2025-04-10 09:00:00",
        resolution_minutes=60*12,
        wcs=WCS(header).dropaxis(2), image_shape=data.shape)
    expected = 147.013
    assert np.allclose(angles, expected)
    assert xpix[0] == -1.0
    assert ypix[0] == -1.0

def test_check_moon_dateobs():
    hdr = fits.getheader(SAMPLE_FITS_PATH_UNCOMPRESSED, ext=1)
    output = check_moon_in_fov(hdr)
    expected = 66.0887
    assert np.allclose(output[1], expected)

wcs = WCS(naxis=2)
wcs.wcs.ctype = "HPLN-AZP", "HPLT-AZP"
wcs.wcs.cunit = "deg", "deg"
wcs.wcs.cdelt = 0.02, 0.02
wcs.wcs.crpix = 1024, 1024
wcs.wcs.crval = 0, 24.75

def test_moon_angular_distance_from_center(): # change import
    # wcs = WCS(fits.getheader(SAMPLE_FITS_PATH_UNCOMPRESSED, ext=1)).dropaxis(2)
    shape = (2048, 2048)
    t_obs = "2026-08-24T16:30:24.766"

    times, ang_sun, fov_half, in_fov, ang_center, xpix, ypix = check_moon_in_fov(
        t_obs, wcs=wcs, image_shape=shape)

    moon = get_body("moon", Time(t_obs)).icrs
    xc = (shape[1] - 1) / 2
    yc = (shape[0] - 1) / 2
    center = wcs.pixel_to_world(xc, yc)
    expected = moon.separation(center).to_value(u.deg)

    assert len(ang_center) == 1
    assert np.isclose(ang_center[0], expected, rtol=1e-6)
