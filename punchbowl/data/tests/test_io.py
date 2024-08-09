import os
from datetime import datetime

import astropy
import numpy as np
import pytest
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS, DistortionLookupTable
from ndcube import NDCube

from punchbowl.data.io import _update_statistics, get_base_file_name, load_ndcube_from_fits, write_ndcube_to_fits
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
        #meta['DATE-OBS'] = str(datetime(2023, 1, 1, 0, 0, 1))
        meta['DATE-OBS'] = str(datetime(2024, 2, 22, 16, 0, 1))
        #20240222163425
        meta['FILEVRSN'] = "1"
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
    w = WCS(naxis=2)
    m = NormalizedMetadata.load_template("PM1", "0")
    m.history.add_now("Test", "does it write?")
    m.history.add_now("Test", "how about twice?")
    m['DESCRPTN'] = 'This is a test!'
    m['CHECKSUM'] = ''
    m['DATASUM'] = ''
    m.delete_section("World Coordinate System")

    sample_data = NDCube(data=np.zeros((2048, 2048),dtype=np.int16), wcs=w, meta=m)

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


def test_filename_base_generation(sample_ndcube):
    cube = sample_ndcube((50, 50))
    actual = get_base_file_name(cube)
    expected = "PUNCH_L0_PM1_20240222160001_v1"
    assert actual == expected


def test_has_typecode():
    meta = NormalizedMetadata.load_template("CFM", "3")
    meta["DATE-OBS"] = str(datetime.now())
    assert "TYPECODE" in meta


def test_load_punchdata_with_history():
    data = np.ones((10, 10), dtype=np.uint16)
    meta = NormalizedMetadata.load_template("CR4", "0")
    meta['DATE-OBS'] = str(datetime.now())
    meta.history.add_now("test", "this is a test!")
    meta.history.add_now("test", "this is a second test!")
    wcs = WCS({"CRVAL1": 0.0,
                     "CRVAL2": 0.0,
                     "CRPIX1": 2047.5,
                     "CRPIX2": 2047.5,
                     "CDELT1": 0.0225,
                     "CDELT2": 0.0225,
                     "CUNIT1": "deg",
                     "CUNIT2": "deg",
                     "CTYPE1": "HPLN-ARC",
                     "CTYPE2": "HPLT-ARC"})
    obj = NDCube(data=data, wcs=wcs, meta=meta)

    assert "OBSCODE" in obj.meta.fits_keys
    file_path = get_base_file_name(obj) + ".fits"
    write_ndcube_to_fits(obj, file_path, overwrite=True)
    reloaded = load_ndcube_from_fits(file_path)
    assert isinstance(reloaded, NDCube)
    assert len(reloaded.meta.history) == 2
    assert reloaded.data.shape == (10, 10)
    assert np.all(reloaded.data == 1)
    os.remove(file_path)


def make_empty_distortion_model(num_bins: int, image: np.ndarray) -> (DistortionLookupTable, DistortionLookupTable):
    """ Create an empty distortion table

    Parameters
    ----------
    num_bins : int
        number of histogram bins in the distortion model, i.e. the size of the distortion model is (num_bins, num_bins)
    image : np.ndarray
        image to create a distortion model for

    Returns
    -------
    (DistortionLookupTable, DistortionLookupTable)
        x and y distortion models
    """
    # make an initial empty distortion model
    r = np.linspace(0, image.shape[0], num_bins + 1)
    c = np.linspace(0, image.shape[1], num_bins + 1)
    r = (r[1:] + r[:-1]) / 2
    c = (c[1:] + c[:-1]) / 2

    err_px, err_py = r, c
    err_x = np.zeros((num_bins, num_bins))
    err_y = np.zeros((num_bins, num_bins))

    cpdis1 = DistortionLookupTable(
        -err_x.astype(np.float32), (0, 0), (err_px[0], err_py[0]), ((err_px[1] - err_px[0]), (err_py[1] - err_py[0]))
    )
    cpdis2 = DistortionLookupTable(
        -err_y.astype(np.float32), (0, 0), (err_px[0], err_py[0]), ((err_px[1] - err_px[0]), (err_py[1] - err_py[0]))
    )
    return cpdis1, cpdis2


def test_write_punchdata_with_distortion():
    data = np.ones((2048, 2048), dtype=np.uint16)
    meta = NormalizedMetadata.load_template("CR4", "1")
    meta['DATE-OBS'] = str(datetime.now())
    meta.history.add_now("test", "this is a test!")
    meta.history.add_now("test", "this is a second test!")
    wcs = WCS({"CRVAL1": 0.0,
                     "CRVAL2": 0.0,
                     "CRPIX1": 2047.5,
                     "CRPIX2": 2047.5,
                     "CDELT1": 0.0225,
                     "CDELT2": 0.0225,
                     "CUNIT1": "deg",
                     "CUNIT2": "deg",
                     "CTYPE1": "HPLN-ARC",
                     "CTYPE2": "HPLT-ARC"})
    cpdis1, cpdis2 = make_empty_distortion_model(100, data)
    wcs.cpdis1 = cpdis1
    wcs.cpdis2 = cpdis2
    obj = NDCube(data=data, wcs=wcs, meta=meta)
    file_path = get_base_file_name(obj) + ".fits"
    write_ndcube_to_fits(obj, file_path, overwrite=True)

    with fits.open(file_path) as hdul:
        assert len(hdul) == 5

    loaded_cube = load_ndcube_from_fits(file_path)
    assert loaded_cube.wcs.has_distortion
