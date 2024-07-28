import os
from datetime import datetime

import astropy
import numpy as np
import pytest
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube

from punchbowl.data.io import (
    _update_statistics,
    construct_wcs_header_fields,
    get_base_file_name,
    load_ndcube_from_fits,
    write_ndcube_to_fits,
)
from punchbowl.data.meta import NormalizedMetadata

TESTDATA_DIR = os.path.dirname(__file__)
SAMPLE_FITS_PATH_UNCOMPRESSED = os.path.join(TESTDATA_DIR, "test_data.fits")
SAMPLE_FITS_PATH_COMPRESSED = os.path.join(TESTDATA_DIR, "test_data.fits")
SAMPLE_WRITE_PATH = os.path.join(TESTDATA_DIR, "write_test.fits")
SAMPLE_OMNIBUS_PATH = os.path.join(TESTDATA_DIR, "omniheader.csv")
SAMPLE_LEVEL_PATH = os.path.join(TESTDATA_DIR, "LevelTest.yaml")
SAMPLE_SPACECRAFT_DEF_PATH = os.path.join(TESTDATA_DIR, "spacecraft.yaml")


@pytest.fixture
def sample_ndcube():
    def _sample_ndcube(shape, code="PM1", level="0"):
        data = np.random.random(shape).astype(np.float32)
        sqrt_abs_data = np.sqrt(np.abs(data))
        uncertainty = StdDevUncertainty(np.interp(sqrt_abs_data, (sqrt_abs_data.min(), sqrt_abs_data.max()), (0,1)).astype(np.float32))
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"
        wcs.wcs.cunit = "deg", "deg"
        wcs.wcs.cdelt = 0.1, 0.1
        wcs.wcs.crpix = 0, 0
        wcs.wcs.crval = 1, 1
        wcs.wcs.cname = "HPC lon", "HPC lat"

        meta = NormalizedMetadata.load_template(code, level)
        meta['DATE-OBS'] = str(datetime(2023, 1, 1, 0, 0, 1))
        return NDCube(data=data, uncertainty=uncertainty, wcs=wcs, meta=meta)
    return _sample_ndcube


def test_write_data(sample_ndcube):
    cube = sample_ndcube((50, 50))
    cube.meta["LEVEL"] = "1"
    cube.meta["TYPECODE"] = "CL"
    cube.meta["OBSRVTRY"] = "1"
    cube.meta["PIPEVRSN"] = "0.1"
    cube.meta["DATE-OBS"] = str(datetime.now())
    cube.meta["DATE-END"] = str(datetime.now())

    write_ndcube_to_fits(cube, SAMPLE_WRITE_PATH)
    assert os.path.isfile(SAMPLE_WRITE_PATH)


def test_generate_data_statistics_from_zeros():
    m = NormalizedMetadata.load_template("PM1", "0")
    m.history.add_now("Test", "does it write?")
    m.history.add_now("Test", "how about twice?")
    m['DESCRPTN'] = 'This is a test!'
    m['CHECKSUM'] = ''
    m['DATASUM'] = ''
    h = m.to_fits_header()

    sample_data = NDCube(data=np.zeros((2048,2048),dtype=np.int16), wcs=WCS(h), meta=m)

    _update_statistics(sample_data)

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


def test_generate_data_statistics(sample_ndcube):
    cube = sample_ndcube((50, 50))
    _update_statistics(cube)

    nonzero_sample_data = cube.data[np.where(cube.data != 0)].flatten()

    assert cube.meta['DATAZER'].value == len(np.where(cube.data == 0)[0])

    assert cube.meta['DATASAT'].value == len(np.where(cube.data >= cube.meta['DSATVAL'].value)[0])

    assert cube.meta['DATAAVG'].value == np.mean(nonzero_sample_data)
    assert cube.meta['DATAMDN'].value == np.median(nonzero_sample_data)
    assert cube.meta['DATASIG'].value == np.std(nonzero_sample_data)

    percentile_percentages = [1, 10, 25, 50, 75, 90, 95, 98, 99]
    percentile_values = np.percentile(nonzero_sample_data, percentile_percentages)

    for percent, value in zip(percentile_percentages, percentile_values):
        assert cube.meta[f'DATAP{percent:02d}'].value == value

    assert cube.meta['DATAMIN'].value == float(cube.data.min())
    assert cube.meta['DATAMAX'].value == float(cube.data.max())


# def test_initial_uncertainty_calculation(sample_punchdata):
#     sample_data = sample_punchdata()
#
#     # Manually call update of uncertainty
#     sample_data = update_initial_uncertainty(sample_data)
#
#     # Check that uncertainty exists and is within range
#     assert sample_data.uncertainty.array.shape == sample_data.data.shape
#     assert sample_data.uncertainty.array.min() >= 0
#     assert sample_data.uncertainty.array.max() <= 1


# def test_read_write_uncertainty_data(sample_punchdata):
#     sample_data = sample_punchdata()
#
#     write_ndcube_to_fits(sample_data, SAMPLE_WRITE_PATH)
#
#     pdata_read_data = load_ndcube_from_fits(SAMPLE_WRITE_PATH)
#
#     with fits.open(SAMPLE_WRITE_PATH) as hdul:
#         fitsio_read_uncertainty = hdul[2].data
#
#     assert sample_data.uncertainty.array.dtype == 'float32'
#     assert pdata_read_data.uncertainty.array.dtype == 'float32'
#     assert fitsio_read_uncertainty.dtype == 'uint8'
#
#     assert (sample_data.uncertainty.array.min() >= 0) and (sample_data.uncertainty.array.max() <= 1)
#     assert (pdata_read_data.uncertainty.array.min() >= 0) and (pdata_read_data.uncertainty.array.max() <= 1)
#     assert (fitsio_read_uncertainty.min() >= 0) and (fitsio_read_uncertainty.max() <= 255)


# def test_uncertainty_bounds(sample_punchdata):
#     sample_data_certain = sample_punchdata()
#     sample_data_certain.uncertainty.array[:, :] = 0
#     write_ndcube_to_fits(sample_data_certain, SAMPLE_WRITE_PATH)
#
#     punchdata_read_data_certain = load_ndcube_from_fits(SAMPLE_WRITE_PATH).uncertainty.array
#     with fits.open(SAMPLE_WRITE_PATH) as hdul:
#         manual_read_data_certain = hdul[2].data
#
#     assert np.all(punchdata_read_data_certain == 0)
#     assert np.all(manual_read_data_certain == 0)
#
#     sample_data_uncertain = sample_punchdata()
#     sample_data_uncertain.uncertainty.array[:, :] = 1
#     write_ndcube_to_fits(sample_data_uncertain, SAMPLE_WRITE_PATH)
#
#     punchdata_read_data_uncertain = load_ndcube_from_fits(SAMPLE_WRITE_PATH).uncertainty.array
#     with fits.open(SAMPLE_WRITE_PATH) as hdul:
#         manual_read_data_uncertain = hdul[2].data
#
#     assert np.all(punchdata_read_data_uncertain == 1)
#     assert np.all(manual_read_data_uncertain == 255)


def test_generate_wcs_metadata(sample_ndcube):
    cube = sample_ndcube((50, 50))
    sample_header = construct_wcs_header_fields(cube)

    assert isinstance(sample_header, astropy.io.fits.Header)


def test_filename_base_generation(sample_ndcube):
    cube = sample_ndcube((50, 50))
    actual = get_base_file_name(cube)
    expected = "PUNCH_L0_PM1_20230101000001"
    assert actual == expected


def test_has_typecode():
    meta = NormalizedMetadata.load_template("CFM", "3")
    meta["DATE-OBS"] = str(datetime.now())
    assert "TYPECODE" in meta


def test_load_punchdata_with_history():
    data = np.ones((10, 10), dtype=np.uint16)
    meta = NormalizedMetadata.load_template("CR4", "1")
    meta['DATE-OBS'] = str(datetime.now())
    meta.history.add_now("test", "this is a test!")
    meta.history.add_now("test", "this is a second test!")
    wcs = WCS(naxis=2)
    obj = NDCube(data=data, wcs=wcs, meta=meta)

    assert "OBSCODE" in obj.meta.fits_keys
    file_path = get_base_file_name(obj) + ".fits"
    write_ndcube_to_fits(obj, file_path, overwrite=True, skip_wcs_conversion=True)
    reloaded = load_ndcube_from_fits(file_path)
    assert isinstance(reloaded, NDCube)
    assert len(reloaded.meta.history) == 2
    assert reloaded.data.shape == (10, 10)
    assert np.all(reloaded.data == 1)
    os.remove(file_path)
