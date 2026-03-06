import os
from datetime import UTC, datetime

import numpy as np
import pytest
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS, DistortionLookupTable
from astropy.wcs.utils import add_stokes_axis_to_wcs
from ndcube import NDCube

from punchbowl.data.meta import NormalizedMetadata
from punchbowl.data.punch_io import (
    CALIBRATION_ANNOTATION,
    _update_statistics,
    check_outlier,
    decode_outliers,
    encode_outliers,
    get_base_file_name,
    load_ndcube_from_fits,
    write_ndcube_to_fits,
    write_ndcube_to_quicklook,
)
from punchbowl.data.wcs import calculate_pc_matrix

TESTDATA_DIR = os.path.dirname(__file__)
SAMPLE_FITS_PATH_UNCOMPRESSED = os.path.join(TESTDATA_DIR, "test_data.fits")
SAMPLE_FITS_PATH_COMPRESSED = os.path.join(TESTDATA_DIR, "test_data.fits")
SAMPLE_OMNIBUS_PATH = os.path.join(TESTDATA_DIR, "omniheader.csv")
SAMPLE_LEVEL_PATH = os.path.join(TESTDATA_DIR, "LevelTest.yaml")
SAMPLE_SPACECRAFT_DEF_PATH = os.path.join(TESTDATA_DIR, "spacecraft.yaml")


@pytest.fixture
def sample_ndcube():
    def _sample_ndcube(shape, code="PM1", level="0", date_obs=None, crota=0):
        data = np.random.random(shape).astype(np.float32)
        uncertainty = StdDevUncertainty(np.sqrt(np.abs(data)))
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"
        wcs.wcs.cunit = "deg", "deg"
        wcs.wcs.cdelt = 0.1, 0.1
        wcs.wcs.crpix = 0, 0
        wcs.wcs.crval = 1, 1
        wcs.wcs.cname = "HPC lon", "HPC lat"
        # For polarized static stray light estimation, which currently excludes northern images
        wcs.wcs.pc = calculate_pc_matrix(crota * np.pi / 180, (0.1, 0.1))

        if level in ["2", "3"] and code[0] == "P":
            wcs = add_stokes_axis_to_wcs(wcs, 2)

        meta = NormalizedMetadata.load_template(code, level)
        meta['DATE-OBS'] = date_obs or str(datetime(2024, 2, 22, 16, 0, 1))
        meta['FILEVRSN'] = "1"

        # Setting these avoids FITSFixedWarnings if these files are written and then read in
        if 'HGLT_OBS' in meta:
            meta['HGLT_OBS'] = 1.6391084786225854
            meta['HGLN_OBS'] = 0.0
            meta['CRLT_OBS'] = 1.6391084786225854
            meta['CRLN_OBS'] = 131.07429379735413
            meta['DSUN_OBS'] = 152011862324.1987
        return NDCube(data=data, uncertainty=uncertainty, wcs=wcs, meta=meta)
    return _sample_ndcube


def test_write_data(sample_ndcube, tmpdir):
    cube = sample_ndcube((50, 50))
    cube.meta["LEVEL"] = "1"
    cube.meta["TYPECODE"] = "CL"
    cube.meta["OBSRVTRY"] = "1"
    cube.meta["DATE-OBS"] = str(datetime.now(UTC))
    cube.meta["DATE-END"] = str(datetime.now(UTC))

    test_path = os.path.join(tmpdir, "test.fits")
    write_ndcube_to_fits(cube, test_path)
    assert os.path.isfile(test_path)

    with fits.open(test_path) as hdul:
        assert hdul[1].header['EXTNAME'] == "PRIMARY DATA ARRAY"
        assert hdul[2].header['EXTNAME'] == "UNCERTAINTY ARRAY"
        assert hdul[3].header['EXTNAME'] == "FILE PROVENANCE"


def test_write_data_jp2(sample_ndcube, tmpdir):
    cube = sample_ndcube((50, 50))
    cube.meta["LEVEL"] = "1"
    cube.meta["TYPECODE"] = "CL"
    cube.meta["OBSRVTRY"] = "1"
    cube.meta["DATE-OBS"] = str(datetime.now(UTC))
    cube.meta["DATE-END"] = str(datetime.now(UTC))

    test_path = os.path.join(tmpdir, "test.jp2")
    write_ndcube_to_quicklook(cube, test_path)
    assert os.path.isfile(test_path)

def test_write_jpeg(sample_ndcube, tmpdir):
    from punchbowl.data.sample import PUNCH_PAM
    cube = sample_ndcube((4096, 4096))
    with fits.open(PUNCH_PAM) as hdul:
        cube.data[...] = hdul[1].data[0]
    cube.meta["LEVEL"] = "1"
    cube.meta["TYPECODE"] = "CL"
    cube.meta["OBSRVTRY"] = "1"
    cube.meta["DATE-OBS"] = str(datetime.now(UTC))
    cube.meta["DATE-END"] = str(datetime.now(UTC))

    test_path = os.path.join(tmpdir, "test.jpeg")
    write_ndcube_to_quicklook(cube, test_path, vmin=1E-15,  vmax=8E-12, annotation="Hi there! I'm {TYPECODE}.", color=True)
    assert os.path.isfile(test_path)


def test_write_data_jp2_with_annotation(sample_ndcube, tmpdir):
    cube = sample_ndcube((2048, 2048))
    cube.meta["LEVEL"] = "1"
    cube.meta["TYPECODE"] = "CL"
    cube.meta["OBSRVTRY"] = "1"
    cube.meta["DATE-OBS"] = str(datetime.now(UTC))
    cube.meta["DATE-END"] = str(datetime.now(UTC))

    test_path = os.path.join(tmpdir, "test.jp2")
    write_ndcube_to_quicklook(cube, test_path, annotation=CALIBRATION_ANNOTATION)
    assert os.path.isfile(test_path)


def test_write_data_jp2_wrong_filename(sample_ndcube, tmpdir):
    cube = sample_ndcube((50, 50))
    cube.meta["LEVEL"] = "1"
    cube.meta["TYPECODE"] = "CL"
    cube.meta["OBSRVTRY"] = "1"
    cube.meta["DATE-OBS"] = str(datetime.now(UTC))
    cube.meta["DATE-END"] = str(datetime.now(UTC))

    test_path = os.path.join(tmpdir, "test.fits")
    with pytest.raises(ValueError):
        write_ndcube_to_quicklook(cube, test_path)


def test_write_data_jp2_wrong_dimensions(sample_ndcube, tmpdir):
    cube = sample_ndcube((2, 50, 50))
    cube.meta["LEVEL"] = "3"
    cube.meta["TYPECODE"] = "CAM"
    cube.meta["DATE-OBS"] = str(datetime.now(UTC))
    cube.meta["DATE-END"] = str(datetime.now(UTC))

    test_path = os.path.join(tmpdir, "test.jp2")
    with pytest.raises(ValueError):
        write_ndcube_to_quicklook(cube, test_path, layer=None)


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

    new_meta = _update_statistics(sample_data)
    sample_data.meta = new_meta

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
    new_meta = _update_statistics(cube)
    cube.meta = new_meta

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
    meta["DATE-OBS"] = str(datetime.now(UTC))
    assert "TYPECODE" in meta


def test_load_punchdata_with_history(tmpdir):
    data = np.ones((10, 10), dtype=np.uint16)
    meta = NormalizedMetadata.load_template("CR4", "0")
    meta['DATE-OBS'] = str(datetime.now(UTC))
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
    file_path = os.path.join(tmpdir, get_base_file_name(obj) + ".fits")
    write_ndcube_to_fits(obj, file_path, overwrite=True)
    reloaded = load_ndcube_from_fits(file_path)
    assert isinstance(reloaded, NDCube)
    assert len(reloaded.meta.history) == 2
    assert reloaded.data.shape == (10, 10)
    assert np.all(reloaded.data == 1)
    os.remove(file_path)


def make_empty_distortion_model(num_bins: int, image: np.ndarray) -> tuple:
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


def test_write_punchdata_with_distortion(tmpdir):
    data = np.ones((2048, 2048), dtype=np.uint16)
    uncertainty = np.zeros_like(data)
    meta = NormalizedMetadata.load_template("CR4", "1")
    meta['DATE-OBS'] = str(datetime.now(UTC))
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
    obj = NDCube(data=data, wcs=wcs, meta=meta, uncertainty=StdDevUncertainty(uncertainty))
    file_path = os.path.join(tmpdir, get_base_file_name(obj) + ".fits")
    write_ndcube_to_fits(obj, file_path, overwrite=True)

    with fits.open(file_path) as hdul:
        assert len(hdul) == 6

    loaded_cube = load_ndcube_from_fits(file_path)
    assert loaded_cube.wcs.has_distortion


def test_uncertainty_inf_roundtrip(sample_ndcube, tmpdir):
    cube = sample_ndcube((50, 50), level="1")
    cube.data[:25] = 0
    cube.uncertainty.array[...] = np.inf
    test_path = os.path.join(tmpdir, "test.fits")
    write_ndcube_to_fits(cube, test_path)
    loaded_cube = load_ndcube_from_fits(test_path)
    assert np.all(np.isinf(loaded_cube.uncertainty.array))


def test_check_outliers(sample_ndcube):
    cube = sample_ndcube((10,10))

    cube.meta["OUTLIER"] = 1
    assert check_outlier(cube) == 1

    cube.meta["OUTLIER"] = 0
    assert check_outlier(cube) == 0

    cube.meta["OUTLIER"] = 15
    assert check_outlier(cube) == 1


def test_encode_outliers(sample_ndcube):
    cube1 = sample_ndcube((10,10))
    cube1.meta["OBSCODE"] = "1"
    cube1.meta["OUTLIER"] = 0

    cube2 = sample_ndcube((10,10))
    cube2.meta["OBSCODE"] = "2"
    cube2.meta["OUTLIER"] = 1

    cube3 = sample_ndcube((10,10))
    cube3.meta["OBSCODE"] = "3"
    cube3.meta["OUTLIER"] = 0

    cube4 = sample_ndcube((10,10))
    cube4.meta["OBSCODE"] = "4"
    cube4.meta["OUTLIER"] = 1

    outlier_code = encode_outliers([cube1, cube2, cube3, cube4])

    assert outlier_code == 20


def test_decode_outliers(sample_ndcube):
    cube = sample_ndcube((10,10))
    cube.meta["OBSCODE"] = "M"
    cube.meta["OUTLIER"] = 20

    outliers = decode_outliers(cube)

    assert outliers["4"] == 1
    assert outliers["3"] == 0
    assert outliers["2"] == 1
    assert outliers["1"] == 0
