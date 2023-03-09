# Core Python imports
from datetime import datetime
import pathlib
import os

# Third party imports
import numpy as np
import pytest
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube
from prefect.logging import disable_run_logger
from pytest import fixture

# punchbowl imports
from punchbowl.data import PUNCHData, NormalizedMetadata
from punchbowl.level1.deficient_pixel import sliding_window, cell_neighbors, mean_example, median_example, remove_deficient_pixels


THIS_DIRECTORY = pathlib.Path(__file__).parent.resolve()


@fixture
def sample_bad_pixel_map(shape: tuple = (2048, 2048), n_bad_pixels: int = 20) -> np.ndarray:
    """
    Generate some random data for testing
    """
    bad_pixel_map = np.ones(shape)

    x_coords = np.fix(np.random.random(n_bad_pixels) * shape[0]).astype(int)
    y_coords = np.fix(np.random.random(n_bad_pixels) * shape[1]).astype(int)

    bad_pixel_map[x_coords, y_coords] = 0

    bad_pixel_map = bad_pixel_map.astype(int)

    uncertainty = StdDevUncertainty(bad_pixel_map)

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLN-AZP", "HPLT-AZP"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.02, 0.02
    wcs.wcs.crpix = 1024, 1024
    wcs.wcs.crval = 0, 24.75

    meta = NormalizedMetadata({"TYPECODE": "CL", "LEVEL": "1", "OBSRVTRY": "0", "DATE-OBS": "2008-01-03 08:57:00"})
    return PUNCHData(data=bad_pixel_map, uncertainty=uncertainty, wcs=wcs, meta=meta)


@fixture
def perfect_pixel_map(shape: tuple = (2048, 2048), n_bad_pixels: int = 20) -> np.ndarray:
    """
    Generate some random data for testing
    """
    bad_pixel_map = np.ones(shape)

    bad_pixel_map = bad_pixel_map.astype(int)

    uncertainty = StdDevUncertainty(bad_pixel_map)

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLN-AZP", "HPLT-AZP"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.02, 0.02
    wcs.wcs.crpix = 1024, 1024
    wcs.wcs.crval = 0, 24.75

    meta = NormalizedMetadata({"TYPECODE": "CL", "LEVEL": "1", "OBSRVTRY": "0", "DATE-OBS": "2008-01-03 08:57:00"})
    return PUNCHData(data=bad_pixel_map, uncertainty=uncertainty, wcs=wcs, meta=meta)


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

    meta = NormalizedMetadata({"TYPECODE": "CL", "LEVEL": "1", "OBSRVTRY": "0", "DATE-OBS": "2008-01-03 08:57:00"})
    return PUNCHData(data=data, uncertainty=uncertainty, wcs=wcs, meta=meta)


@pytest.mark.prefect_test
def test_remove_deficient_pixels(sample_punchdata, sample_bad_pixel_map):
    """
    Test the remove_deficient_pixels prefect flow using a test harness, providing a filename
    """
    with disable_run_logger():
        flagged_punchdata = remove_deficient_pixels.fn(sample_punchdata,
                                                       sample_bad_pixel_map)

        assert isinstance(flagged_punchdata, PUNCHData) 
        #assert np.all(flagged_punchdata.uncertainty[np.where(sample_pixel_map == 1)].array == np.inf)

@pytest.mark.prefect_test_2
def test_nan_input(sample_punchdata, sample_bad_pixel_map):
    """
    The module output is tested when NaN data points are included in the input PUNCHData object. Test for no errors.
    """

    input_data = sample_punchdata
    input_data.data[42, 42] = np.nan

    with disable_run_logger():
        flagged_punchdata = remove_deficient_pixels.fn(input_data, sample_bad_pixel_map)

        assert isinstance(flagged_punchdata, PUNCHData)
        assert np.all(flagged_punchdata.uncertainty[np.where(sample_bad_pixel_map == 1)].array == np.inf)



@pytest.mark.prefect_test
def test_data_loading(sample_punchdata, perfect_pixel_map):
    """
    A specific observation is provided. The module loads it as a PUNCHData object. 
    No bad data points, in same as out. uncertainty should be the same in and out.
    """
    with disable_run_logger():
        deficient_punchdata = remove_deficient_pixels.fn(sample_punchdata, perfect_pixel_map)

    assert isinstance(deficient_punchdata, PUNCHData)
    assert np.all(deficient_punchdata.data == sample_punchdata.data)
    assert np.all(deficient_punchdata.uncertainty.array == sample_punchdata.uncertainty.array)


@pytest.mark.prefect_test
def test_artificial_pixel_map(sample_punchdata, sample_bad_pixel_map):
    """
    A known artificial bad pixel map is ingested. The output flags are tested against the input map.
    """

    with disable_run_logger():
        flagged_punchdata = remove_deficient_pixels.fn(sample_punchdata, sample_bad_pixel_map)

        assert isinstance(flagged_punchdata, PUNCHData)
        assert np.all(flagged_punchdata.uncertainty[np.where(sample_bad_pixel_map == 1)].array == np.inf)


# @pytest.mark.prefect_test
# def test_flag_task_filename(sample_punchdata):
#    """
#    Test the flag_task prefect flow using a test harness, providing a filename
#    """
#    with disable_run_logger():
#        flagged_punchdata = remove_deficient_pixels.fn(sample_punchdata,
#                                         bad_pixel_filename=os.path.join(str(THIS_DIRECTORY),
#                                        'data/PUNCH_L1_DP0_20080103085700.fits'))
#        assert isinstance(flagged_punchdata, PUNCHData)
#        assert np.all(flagged_punchdata.uncertainty[np.where(sample_bad_pixel_map == 1)].array == np.inf)



@pytest.mark.prefect_test
def test_flag_task_nofilename(sample_punchdata):
    """
    Test the flag_task prefect flow using a test harness, failing to provide a filename - an error should occur
    """
    with disable_run_logger():
        with pytest.raises(Exception):
            flagged_punchdata = remove_deficient_pixels.fn(sample_punchdata)