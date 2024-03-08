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
from numpy.linalg import inv
from pytest import fixture
from sunpy import sun
from sunpy.coordinates import frames

from punchbowl.data import (
    History,
    HistoryEntry,
    MetaField,
    NormalizedMetadata,
    PUNCHData,
    calculate_helio_wcs_from_celestial,
    load_spacecraft_def,
    load_trefoil_wcs,
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
def sample_punchdata_clear():
    """
    Generate a sample PUNCHData object for testing
    """

    def _sample_punchdata_clear(shape=(50, 50)):
        data = np.random.random(shape).astype(np.float32)
        uncertainty = StdDevUncertainty(np.sqrt(np.abs(data)))
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"
        wcs.wcs.cunit = "deg", "deg"
        wcs.wcs.cdelt = 0.1, 0.1
        wcs.wcs.crpix = 0, 0
        wcs.wcs.crval = 1, 1
        wcs.wcs.cname = "HPC lon", "HPC lat"

        meta = NormalizedMetadata.load_template("CR1", "0")
        meta['DATE-OBS'] = str(datetime(2023, 1, 1, 0, 0, 1))
        return PUNCHData(data=data, uncertainty=uncertainty, wcs=wcs, meta=meta)

    return _sample_punchdata_clear


@fixture
def sample_punchdata_list(sample_punchdata):
    """
    Generate a list of sample PUNCHData objects for testing
    """
    sample_pd1 = sample_punchdata()
    sample_pd2 = sample_punchdata()
    return [sample_pd1, sample_pd2]

@fixture
def sample_punchdata_triplet(sample_punchdata):
    """
    Generate a list of sample PUNCHData objects for testing polarization resolving
    """

    sample_pd1 = sample_punchdata()
    sample_pd2 = sample_punchdata()
    sample_pd3 = sample_punchdata()
    return [sample_pd1, sample_pd2, sample_pd3]


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


def test_normalizedmetadata_from_fits_header():
    m = NormalizedMetadata.load_template("PM1", "0")
    m['DESCRPTN'] = 'This is a test!'
    m.history.add_entry(HistoryEntry(datetime(2023, 10, 30, 12, 20), "test", "test comment"))
    m.history.add_entry(HistoryEntry(datetime(2023, 10, 30, 12, 20), "test2", "test comment"))
    h = m.to_fits_header()

    recovered = NormalizedMetadata.from_fits_header(h)

    assert recovered == m


# def test_generate_level3_data_product(tmpdir):
#     m = NormalizedMetadata.load_template("PTM", "3")
#     m['DATE-OBS'] = datetime.utcnow().isoformat()
#     h = m.to_fits_header()
#
#     path = os.path.join(tmpdir, "from_fits_test.fits")
#     d = PUNCHData(np.ones((2, 4096, 4096), dtype=np.float32), WCS(h), m)
#     d.write(path)
#
#     loaded = PUNCHData.from_fits(path)
#     loaded.meta['LATPOLE'] = 0.0
#
#     assert loaded.meta == m


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


def test_wcs_many_point_2d_check():
    m = NormalizedMetadata.load_template("CTM", "2")
    # m['DATE-OBS'] = str(datetime.utcnow().isoformat())
    date_obs = Time("2024-01-01T00:00:00", format='isot', scale='utc')
    m['DATE-OBS'] = str(date_obs)
    # date_obs = Time(m['DATE-OBS'].value, format='isot', scale='utc')
    sun_radec = get_sun(date_obs)
    m['CRVAL1A'] = sun_radec.ra.to(u.deg).value
    m['CRVAL2A'] = sun_radec.dec.to(u.deg).value
    h = m.to_fits_header()
    d = PUNCHData(np.ones((4096, 4096), dtype=np.float32), WCS(h, key='A'), m)

    # we're at the center of the Earth so let's try that
    test_loc = EarthLocation.from_geocentric(0, 0, 0, unit=u.m)
    test_gcrs = SkyCoord(test_loc.get_gcrs(date_obs))

    wcs_celestial = d.wcs

    wcs_helio, _ = calculate_helio_wcs_from_celestial(wcs_celestial, date_obs, d.data.shape)

    npoints = 20
    input_coords = np.stack([
                             np.linspace(0, 4096, npoints).astype(int),
                             np.linspace(0, 4096, npoints).astype(int)], axis=1)

    points_celestial = wcs_celestial.all_pix2world(input_coords, 0)
    points_helio = wcs_helio.all_pix2world(input_coords, 0)

    output_coords = []

    for c_pix, c_celestial, c_helio in zip(input_coords, points_celestial, points_helio):
        # skycoord_helio = SkyCoord(c_helio[1] * u.deg, c_helio[2] * u.deg,
        #                           frame=frames.Helioprojective,
        #                           obstime=date_obs,
        #                           observer=test_gcrs)
        skycoord_celestial = SkyCoord(c_celestial[0] * u.deg, c_celestial[1] * u.deg,
                                      frame=GCRS,
                                      obstime=date_obs,
                                      observer=test_gcrs,
                                      obsgeoloc=test_gcrs.cartesian,
                                      obsgeovel=test_gcrs.velocity.to_cartesian(),
                                      distance=test_gcrs.hcrs.distance
                                      )

        intermediate = skycoord_celestial.transform_to(frames.Helioprojective)
        output_coords.append(wcs_helio.all_world2pix(intermediate.data.lon.to(u.deg).value,
                                                     intermediate.data.lat.to(u.deg).value, 0))

    output_coords = np.array(output_coords)
    distances = np.linalg.norm(input_coords - output_coords, axis=1)
    assert np.mean(distances) < 0.1


def test_wcs_many_point_3d_check():
    m = NormalizedMetadata.load_template("PSN", "3")
    date_obs = Time("2024-01-01T00:00:00", format='isot', scale='utc')
    m['DATE-OBS'] = str(date_obs)
    sun_radec = get_sun(date_obs)
    m['CRVAL2A'] = sun_radec.ra.to(u.deg).value
    m['CRVAL3A'] = sun_radec.dec.to(u.deg).value
    h = m.to_fits_header()
    d = PUNCHData(np.ones((2, 4096, 4096), dtype=np.float32), WCS(h, key='A'), m)

    # we're at the center of the Earth so let's try that
    test_loc = EarthLocation.from_geocentric(0, 0, 0, unit=u.m)
    test_gcrs = SkyCoord(test_loc.get_gcrs(date_obs))

    wcs_celestial = d.wcs
    wcs_helio, _ = calculate_helio_wcs_from_celestial(wcs_celestial, date_obs, d.data.shape)

    print("d", d.wcs)
    print("celestial", wcs_celestial)
    npoints = 20
    input_coords = np.stack([np.ones(npoints, dtype=int),
                             np.linspace(0, 4096, npoints).astype(int),
                             np.linspace(0, 4096, npoints).astype(int)], axis=1)

    points_celestial = wcs_celestial.all_pix2world(input_coords, 0)
    points_helio = wcs_helio.all_pix2world(input_coords, 0)

    output_coords = []
    for c_pix, c_celestial, c_helio in zip(input_coords, points_celestial, points_helio):
        skycoord_celestial = SkyCoord(c_celestial[1] * u.deg, c_celestial[2] * u.deg,
                                      frame=GCRS,
                                      obstime=date_obs,
                                      observer=test_gcrs,
                                      obsgeoloc=test_gcrs.cartesian,
                                      obsgeovel=test_gcrs.velocity.to_cartesian(),
                                      distance=test_gcrs.hcrs.distance
                                      )

        intermediate = skycoord_celestial.transform_to(frames.Helioprojective)
        output_coords.append(wcs_helio.all_world2pix(1,
                                                     intermediate.data.lon.to(u.deg).value,
                                                     intermediate.data.lat.to(u.deg).value, 0))

    output_coords = np.array(output_coords)
    distances = np.linalg.norm(input_coords - output_coords, axis=1)
    assert np.mean(distances) < 2.0  # TODO: figure out why the value has to be high


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


def test_has_typecode():
    meta = NormalizedMetadata.load_template("CFM", "3")
    meta["DATE-OBS"] = str(datetime.now())
    assert "TYPECODE" in meta
