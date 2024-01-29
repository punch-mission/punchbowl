import os
from datetime import datetime
from collections import OrderedDict

import astropy
import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import GCRS, ICRS, SkyCoord
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.time import Time
from astropy.wcs import WCS
from ndcube import NDCube
from numpy.linalg import inv
from pytest import fixture
from sunpy.coordinates import frames, sun

from punchbowl.data import (
    History,
    HistoryEntry,
    MetaField,
    NormalizedMetadata,
    PUNCHData,
    load_spacecraft_def,
    load_trefoil_wcs,
)
from punchbowl.exceptions import InvalidDataError

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


def test_normalizedmetadata_get_keys():
    result = NormalizedMetadata.load_template("CB1",
                                              omniheader_path=SAMPLE_OMNIBUS_PATH,
                                              level_spec_path=SAMPLE_LEVEL_PATH,
                                              spacecraft_def_path=SAMPLE_SPACECRAFT_DEF_PATH)
    keys = result.fits_keys
    assert len(keys) == 5

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
    wcs.wcs.ctype = "STOKES", "HPLT-TAN", "HPLN-TAN"
    wcs.wcs.cunit = "", "deg", "deg"
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

    def _sample_punchdata(shape=(50, 50)):
        data = np.random.random(shape).astype(np.float32)
        uncertainty = StdDevUncertainty(np.sqrt(np.abs(data)))
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"
        wcs.wcs.cunit = "deg", "deg"
        wcs.wcs.cdelt = 0.1, 0.1
        wcs.wcs.crpix = 0, 0
        wcs.wcs.crval = 1, 1
        wcs.wcs.cname = "HPC lon", "HPC lat"

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


def test_generate_data_statistics_from_zeros(sample_punchdata):
    m = NormalizedMetadata.load_template("PM1", "0")
    m.history.add_now("Test", "does it write?")
    m.history.add_now("Test", "how about twice?")
    m['DESCRPTN'] = 'This is a test!'
    m['CHECKSUM'] = ''
    m['DATASUM'] = ''
    h = m.to_fits_header()

    sample_data = PUNCHData(data=np.zeros((2048,2048),dtype=np.int16), wcs=WCS(h), meta=m)

    sample_data._update_statistics()

    assert sample_data.meta['DATAZER'].value == 2048*2048

    assert sample_data.meta['DATASAT'].value == 0

    assert sample_data.meta['DATAAVG'].value == -999.0
    assert sample_data.meta['DATAMDN'].value == -999.0
    assert sample_data.meta['DATASIG'].value == -999.0

    percentile_percentages = [1, 10, 25, 50, 75, 90, 95, 98, 99]
    percentile_values = [-999.0 for _ in percentile_percentages]

    for percent, value in zip(percentile_percentages, percentile_values):
        assert sample_data.meta[f'DATAP{percent:02d}'].value == value

    assert sample_data.meta['DATAMIN'].value == 0.0
    assert sample_data.meta['DATAMAX'].value == 0.0

def test_generate_data_statistics(sample_punchdata):
    sample_data = sample_punchdata()

    sample_data._update_statistics()

    nonzero_sample_data = sample_data.data[np.where(sample_data.data != 0)].flatten()

    assert sample_data.meta['DATAZER'].value == len(np.where(sample_data.data == 0)[0])

    assert sample_data.meta['DATASAT'].value == len(np.where(sample_data.data >= sample_data.meta['DSATVAL'].value)[0])

    assert sample_data.meta['DATAAVG'].value == np.mean(nonzero_sample_data)
    assert sample_data.meta['DATAMDN'].value == np.median(nonzero_sample_data)
    assert sample_data.meta['DATASIG'].value == np.std(nonzero_sample_data)

    percentile_percentages = [1, 10, 25, 50, 75, 90, 95, 98, 99]
    percentile_values = np.percentile(nonzero_sample_data, percentile_percentages)

    for percent, value in zip(percentile_percentages, percentile_values):
        assert sample_data.meta[f'DATAP{percent:02d}'].value == value

    assert sample_data.meta['DATAMIN'].value == float(sample_data.data.min())
    assert sample_data.meta['DATAMAX'].value == float(sample_data.data.max())


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


def test_generate_wcs_metadata(sample_punchdata):
    sample_data = sample_punchdata()
    sample_header = sample_data.construct_wcs_header_fields()

    assert isinstance(sample_header, astropy.io.fits.Header)


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
    m['DATE-OBS'] = datetime.utcnow().isoformat()
    m.history.add_now("Test", "does it write?")
    m.history.add_now("Test", "how about twice?")
    m['DESCRPTN'] = 'This is a test!'
    m['CHECKSUM'] = ''
    m['DATASUM'] = ''
    h = m.to_fits_header()

    path = os.path.join(tmpdir, "from_fits_test.fits")
    d = PUNCHData(data=np.tile(np.arange(2048, dtype=np.int16), (2048, 1)), wcs=WCS(h), meta=m)
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


def test_generate_level3_data_product(tmpdir):
    m = NormalizedMetadata.load_template("PTM", "3")
    m['DATE-OBS'] = datetime.utcnow().isoformat()
    h = m.to_fits_header()

    path = os.path.join(tmpdir, "from_fits_test.fits")
    d = PUNCHData(np.ones((2, 4096, 4096), dtype=np.float32), WCS(h), m)
    d.write(path)

    loaded = PUNCHData.from_fits(path)
    loaded.meta['LATPOLE'] = 0.0

    assert loaded.meta == m


def test_sun_location():
    time_current = Time(datetime.utcnow())

    skycoord_sun = astropy.coordinates.get_sun(Time(datetime.utcnow()))

    skycoord_origin = SkyCoord(0*u.deg, 0*u.deg,
                              frame=frames.Helioprojective,
                              obstime=time_current,
                              observer='earth')

    with frames.Helioprojective.assume_spherical_screen(skycoord_origin.observer):
        skycoord_origin_celestial = skycoord_origin.transform_to(GCRS)

    with frames.Helioprojective.assume_spherical_screen(skycoord_origin.observer):
        assert skycoord_origin_celestial.separation(skycoord_sun) < 1 * u.arcsec
        assert skycoord_origin.separation(skycoord_sun) < 1 * u.arcsec


def test_wcs_center_transformation():
    m = NormalizedMetadata.load_template("CTM", "3")
    m['DATE-OBS'] = datetime.utcnow().isoformat()
    h = m.to_fits_header()
    d = PUNCHData(np.ones((4096, 4096), dtype=np.float32), WCS(h), m)

    # Update the WCS
    updated_header = d.construct_wcs_header_fields()[:-3]

    header_helio = astropy.io.fits.Header()
    header_celestial = astropy.io.fits.Header()

    for key in updated_header.keys():
        if key[-1] == 'A':
            header_celestial[key[:-1]] = updated_header[key]
        else:
            header_helio[key] = updated_header[key]

    wcs_helio = WCS(header_helio)
    wcs_celestial = WCS(header_celestial)

    coords_center = np.array([[0,2047.5, 2047.5]])
    center_helio = wcs_helio.wcs_pix2world(coords_center, 0)
    center_celestial = wcs_celestial.wcs_pix2world(coords_center, 0)

    skycoord_helio = SkyCoord(center_helio[0,1]*u.deg, center_helio[0,2]*u.deg,
                              frame=frames.Helioprojective,
                              obstime=m['DATE-OBS'].value,
                              observer='earth')
    skycoord_celestial = SkyCoord(center_celestial[0,1]*u.deg, center_celestial[0,2]*u.deg,
                                  frame=GCRS,
                                  obstime=m['DATE-OBS'].value)

    with frames.Helioprojective.assume_spherical_screen(skycoord_helio.observer):
        skycoord_celestial_transform = skycoord_helio.transform_to(GCRS)

    with frames.Helioprojective.assume_spherical_screen(skycoord_helio.observer):
        assert skycoord_celestial.separation(skycoord_helio) < 1 * u.arcsec
        assert skycoord_celestial_transform.separation(skycoord_helio) < 1 * u.arcsec


def test_wcs_point_transformation():
    m = NormalizedMetadata.load_template("CTM", "3")
    m['DATE-OBS'] = datetime.utcnow().isoformat()
    h = m.to_fits_header()
    d = PUNCHData(np.ones((4096, 4096), dtype=np.float32), WCS(h), m)

    # Update the WCS
    updated_header = d.construct_wcs_header_fields()[:-3]

    header_helio = astropy.io.fits.Header()
    header_celestial = astropy.io.fits.Header()

    for key in updated_header.keys():
        if key[-1] == 'A':
            header_celestial[key[:-1]] = updated_header[key]
        else:
            header_helio[key] = updated_header[key]

    wcs_helio = WCS(header_helio)
    wcs_celestial = WCS(header_celestial)

    npoints = 20
    coords_points = np.stack([np.zeros(npoints,dtype=int),
                              (np.random.random(npoints)*4095).astype(int),
                              (np.random.random(npoints)*4095).astype(int)], axis=1)

    points_helio = wcs_helio.all_pix2world(coords_points, 0)
    points_celestial = wcs_celestial.all_pix2world(coords_points, 0)

    coord_center = np.array([[0,2047.5, 2047.5]])
    center_helio = wcs_helio.wcs_pix2world(coord_center, 0)

    center_skycoord_helio = SkyCoord(center_helio[0,1]*u.deg, center_helio[0,2]*u.deg,
                              frame=frames.Helioprojective,
                              obstime=m['DATE-OBS'].value,
                              observer='earth')

    for i in np.arange(points_helio.shape[0]):
        skycoord_helio = SkyCoord(points_helio[i, 1] * u.deg, points_helio[i, 2] * u.deg,
                                  frame=frames.Helioprojective,
                                  obstime=m['DATE-OBS'].value,
                                  observer='earth')
        skycoord_celestial = SkyCoord(points_celestial[i, 1] * u.deg, points_celestial[i, 2] * u.deg,
                                      frame=GCRS,
                                      obstime=m['DATE-OBS'].value,
                                      observer='earth')

        with frames.Helioprojective.assume_spherical_screen(SkyCoord(center_skycoord_helio.observer)):
            skycoord_celestial_transform = skycoord_helio.transform_to(GCRS)

        with frames.Helioprojective.assume_spherical_screen(SkyCoord(center_skycoord_helio.observer)):
            assert skycoord_celestial.separation(skycoord_helio) < 25 * u.deg
            assert skycoord_celestial_transform.separation(skycoord_helio) < 1 * u.arcsec


def test_pc_matrix_rotation():
    dateobs = '2023-08-17T00:08:31.006'
    p_angle = sun.P(time=dateobs)

    pc_helio = np.array([[0.959583580000, -0.281423780000],[0.281423780000, 0.959583580000]])
    pc_celestial = np.array([[0.815617440000, -0.578591590000],[0.578591590000, 0.815617440000]])

    rotation_matrix = np.array([[np.cos(p_angle), -1 * np.sin(p_angle)], [np.sin(p_angle), np.cos(p_angle)]])

    rotation_matrix_computed = np.matmul(pc_helio, inv(pc_celestial))
    p_angle_computed = (np.arcsin(-1* rotation_matrix_computed[1,0])*u.rad).to(u.deg)

    pc_celestial_computed = np.matmul(pc_helio, rotation_matrix)

    assert abs(p_angle_computed - p_angle) < 5 * u.deg


def test_axis_ordering():
    data = np.random.random((2,256,256))
    hdu_data = fits.ImageHDU(data=data)

    assert hdu_data.header['NAXIS1'] == data.shape[2]
    assert hdu_data.header['NAXIS2'] == data.shape[1]
    assert hdu_data.header['NAXIS3'] == data.shape[0]


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


def test_load_trefoil_wcs():
    trefoil_wcs, trefoil_shape = load_trefoil_wcs()
    assert trefoil_shape == (4096, 4096)
    assert isinstance(trefoil_wcs, WCS)
