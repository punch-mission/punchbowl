# Core Python imports

# Third party imports
import numpy as np
import pytest
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube
from prefect.logging import disable_run_logger
from pytest import fixture

# punchbowl imports
from punchbowl.data import PUNCHData
from punchbowl.level1.flagging import flag_punchdata, flag_task


@fixture
def sample_pixel_map(shape: tuple = (2048, 2048), n_bad_pixels: int = 20) -> np.ndarray:
    """
    Generate some random data for testing
    """

    bad_pixel_map = np.zeros(shape)

    x_coords = np.fix(np.random.random(n_bad_pixels) * shape[0]).astype(int)
    y_coords = np.fix(np.random.random(n_bad_pixels) * shape[1]).astype(int)

    bad_pixel_map[x_coords, y_coords] = 1

    bad_pixel_map = bad_pixel_map.astype(bool)

    return bad_pixel_map


@fixture
def sample_punchdata(shape: tuple = (2048, 2048)) -> PUNCHData:
    """
    Generate a sample PUNCHData object for testing
    """

    data = np.random.random(shape)
    uncertainty = StdDevUncertainty(np.sqrt(np.abs(data)))

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLN-AZP", "HPLT-AZP"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.02, 0.02
    wcs.wcs.crpix = 1024, 1024
    wcs.wcs.crval = 0, 24.75

    nd_obj = NDCube(data=data, uncertainty=uncertainty, wcs=wcs)

    nd_obj.meta["TYPECODE"] = 'CL'
    nd_obj.meta["LEVEL"] = '1'
    nd_obj.meta["OBSRVTRY"] = '0'
    # TODO - date-obs lowercase convention here? data.py demands lowercase
    nd_obj.meta["date-obs"] = '2008-01-03 08:57:00'

    return PUNCHData(nd_obj)


def test_data_loading(sample_punchdata):
    """
    A specific observation is provided. The module loads it as a PUNCHData object. Test for no errors.
    """

    flagged_punchdata = flag_punchdata(sample_punchdata)

    assert isinstance(flagged_punchdata, PUNCHData)
    assert np.all(flagged_punchdata.data == sample_punchdata.data)
    assert np.all(flagged_punchdata.uncertainty == sample_punchdata.uncertainty)


def test_nan_input(sample_punchdata, sample_pixel_map):
    """
    The module output is tested when NaN data points are included in the input PUNCHData object. Test for no errors.
    """

    input_data = sample_punchdata

    input_data.data[42, 42] = np.nan

    flagged_punchdata = flag_punchdata(input_data, sample_pixel_map)

    assert isinstance(flagged_punchdata, PUNCHData)
    assert np.all(flagged_punchdata.data[np.where(sample_pixel_map == 1)] == 0)
    assert np.all(flagged_punchdata.uncertainty[np.where(sample_pixel_map == 1)].array == np.inf)


def test_artificial_pixel_map(sample_punchdata, sample_pixel_map):
    """
    A known artificial bad pixel map is ingested. The output flags are tested against the input map.
    """

    flagged_punchdata = flag_punchdata(sample_punchdata, sample_pixel_map)

    assert isinstance(flagged_punchdata, PUNCHData)
    assert np.all(flagged_punchdata.data[np.where(sample_pixel_map == 1)] == 0)
    assert np.all(flagged_punchdata.uncertainty[np.where(sample_pixel_map == 1)].array == np.inf)


@pytest.mark.prefect_test
def test_flag_task_filename(sample_punchdata):
    """
    Test the flag_task prefect flow using a test harness, providing a filename
    """
    with disable_run_logger():
        flagged_punchdata = flag_task.fn(sample_punchdata, bad_pixel_filename='data/PUNCH_L1_DP0_20080103085700.fits')

        assert isinstance(flagged_punchdata, PUNCHData)
        assert np.all(flagged_punchdata.data[np.where(sample_pixel_map == 1)] == 0)
        assert np.all(flagged_punchdata.uncertainty[np.where(sample_pixel_map == 1)].array == np.inf)

@pytest.mark.prefect_test
def test_flag_task_nofilename(sample_punchdata):
    """
    Test the flag_task prefect flow using a test harness, generating a filename
    """
    with disable_run_logger():
        flagged_punchdata = flag_task.fn(sample_punchdata)

        assert isinstance(flagged_punchdata, PUNCHData)
        assert np.all(flagged_punchdata.data[np.where(sample_pixel_map == 1)] == 0)
        assert np.all(flagged_punchdata.uncertainty[np.where(sample_pixel_map == 1)].array == np.inf)