# Core Python imports
import pathlib

# Third party imports
import numpy as np
import pytest
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube
from prefect.logging import disable_run_logger

# punchbowl imports
from punchbowl.data import NormalizedMetadata
from punchbowl.level1.deficient_pixel import remove_deficient_pixels_task
from punchbowl.level2.bright_structure import run_zspike, identify_bright_structures_task


@pytest.fixture()
def sample_bad_pixel_map(shape: tuple = (2048, 2048), n_bad_pixels: int = 20) -> NDCube:
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
    return NDCube(data=bad_pixel_map, uncertainty=uncertainty, wcs=wcs, meta=meta)

@pytest.fixture()
def sample_punchdata(shape: tuple = (5, 2048, 2048)) -> NDCube:
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
    return NDCube(data=data, uncertainty=uncertainty, wcs=wcs, meta=meta)


@pytest.fixture()
def even_sample_punchdata(shape: tuple = (6, 2048, 2048)) -> NDCube:
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
    return NDCube(data=data, uncertainty=uncertainty, wcs=wcs, meta=meta)


@pytest.fixture()
def sample_zero_punchdata(shape: tuple = (5, 2048, 2048)) -> NDCube:
    """
    Generate a sample PUNCHData object for testing
    """

    data = np.zeros(shape)
    uncertainty = StdDevUncertainty(np.sqrt(np.abs(data)))

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLN-AZP", "HPLT-AZP"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.02, 0.02
    wcs.wcs.crpix = 1024, 1024
    wcs.wcs.crval = 0, 24.75

    meta = NormalizedMetadata({"TYPECODE": "CL", "LEVEL": "1", "OBSRVTRY": "0", "DATE-OBS": "2008-01-03 08:57:00"})
    return NDCube(data=data, uncertainty=uncertainty, wcs=wcs, meta=meta)


@pytest.fixture()
def one_bright_point_sample_punchdata(shape: tuple = (7, 2048, 2048)) -> NDCube:
    """
    Generate a sample PUNCHData object for testing
    """
    x_interest = 200
    y_interest = 200
    data = np.random.random(shape)

    # add a bright point
    data[3, x_interest, y_interest]=1000
    uncertainty = StdDevUncertainty(np.sqrt(np.abs(data)))


    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLN-AZP", "HPLT-AZP"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.02, 0.02
    wcs.wcs.crpix = 1024, 1024
    wcs.wcs.crval = 0, 24.75

    meta = NormalizedMetadata({"TYPECODE": "CL", "LEVEL": "1", "OBSRVTRY": "0", "DATE-OBS": "2008-01-03 08:57:00"})
    return NDCube(data=data, uncertainty=uncertainty, wcs=wcs, meta=meta)

@pytest.fixture()
def two_bright_point_sample_punchdata(shape: tuple = (7, 2048, 2048)) -> NDCube:
    """
    Generate a sample PUNCHData object for testing
    """
    x_interest = 200
    y_interest = 200
    data = np.random.random(shape)

    # add two bright points in a row
    data[3, x_interest, y_interest]=1000
    data[4, x_interest, y_interest]=1000
    uncertainty = StdDevUncertainty(np.sqrt(np.abs(data)))


    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLN-AZP", "HPLT-AZP"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.02, 0.02
    wcs.wcs.crpix = 1024, 1024
    wcs.wcs.crval = 0, 24.75

    meta = NormalizedMetadata({"TYPECODE": "CL", "LEVEL": "1", "OBSRVTRY": "0", "DATE-OBS": "2008-01-03 08:57:00"})
    return NDCube(data=data, uncertainty=uncertainty, wcs=wcs, meta=meta)


def test_valid_data_and_uncertainty(sample_punchdata: NDCube):
    with pytest.raises(Exception):
        # Call find_spikes with valid data and uncertainty
        result = run_zspike(sample_punchdata.data, sample_punchdata.uncertainty.array)
        # Perform assertions on the result
        # For example, check the shape or the datatype of the result
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_punchdata.data.shape

def test_zero_threshold(sample_punchdata: NDCube):
    # test the thresholds are zero
    threshold = 0
    result = run_zspike(sample_punchdata.data,
                        sample_punchdata.uncertainty.array,
                        threshold=threshold,
                        diff_method='abs')
    assert np.sum(result) == 0

    result2 = run_zspike(sample_punchdata.data,
                         sample_punchdata.uncertainty.array,
                         threshold=threshold,
                         diff_method='sigma')
    assert np.sum(result) == 0
    #assert np.array_equal(result_abs, result_sigma)


def test_diff_methods(sample_zero_punchdata: NDCube):
    result_abs = run_zspike(sample_zero_punchdata.data, sample_zero_punchdata.uncertainty.array, diff_method='abs')
    result_sigma = run_zspike(sample_zero_punchdata.data, sample_zero_punchdata.uncertainty.array, diff_method='sigma')
    assert np.array_equal(result_abs, result_sigma)


def test_different_parameters(sample_punchdata: NDCube):
    required_yes = 1
    veto_limit = 1
    dilation = 0
    result = run_zspike(sample_punchdata.data, sample_punchdata.uncertainty.array, required_yes=required_yes, veto_limit=veto_limit, dilation=dilation)
    assert result.shape == np.shape(sample_punchdata.data[0,:,:])

#####
def test_raise_error_insufficient_frames(sample_bad_pixel_map: NDCube):
    # creates a raise error as only a 2d array is passed in

    with pytest.raises(ValueError):
        #sample_data.write(SAMPLE_WRITE_PATH)
        run_zspike(sample_bad_pixel_map.data,
                   sample_bad_pixel_map.uncertainty.array)


def test_raise_error(even_sample_punchdata: NDCube):
    # creates a raise error as an even array is passed in

    with pytest.raises(ValueError):
        #sample_data.write(SAMPLE_WRITE_PATH)
        run_zspike(even_sample_punchdata.data,
                   even_sample_punchdata.uncertainty.array)


def test_raise_no_error(even_sample_punchdata: NDCube):
    # does not create a raise error as an even array is passed in and an index of interest

    result=run_zspike(even_sample_punchdata.data,
                      even_sample_punchdata.uncertainty.array,
                      index_of_interest=3)

    assert isinstance(result, np.ndarray)
    assert result.shape == np.shape(even_sample_punchdata.data[0,:,:])

def test_single_bright_point(sample_punchdata: NDCube):
    # test passes with single bright point
    test_data=sample_punchdata.data
    test_uncertainty=sample_punchdata.uncertainty.array
    test_data.data[3, 200, 200]=1000
    # add spike

    result=run_zspike(test_data, test_uncertainty)

    assert result.shape == np.shape(sample_punchdata.data[0,:,:])
    assert isinstance(result, np.ndarray)

def test_single_bright_point_2(one_bright_point_sample_punchdata: NDCube):

    x_interest = 200
    y_interest = 200

    result_2 = run_zspike(one_bright_point_sample_punchdata.data,
                          one_bright_point_sample_punchdata.uncertainty.array,
                          diff_method='abs',
                          threshold=3,
                          required_yes=1,
                          veto_limit=1)

    # test cell of interest is set to 'True'
    assert result_2[x_interest, y_interest] == True

    # test other cells are set to 'False'
    assert result_2[x_interest+1, y_interest] == False


def test_veto(two_bright_point_sample_punchdata: NDCube):
    # test works with one vote
    x_interest = 200
    y_interest = 200

    result_2 = run_zspike(two_bright_point_sample_punchdata.data,
                          two_bright_point_sample_punchdata.uncertainty.array,
                          diff_method='abs',
                          threshold=3,
                          required_yes=1,
                          veto_limit=1)

    # test cell of interest is set to 'False' with veto
    assert result_2[x_interest, y_interest] == True

    two_bright_point_sample_punchdata.uncertainty.array[:, :, :] = 0


    result_3 = run_zspike(two_bright_point_sample_punchdata.data,
                          two_bright_point_sample_punchdata.uncertainty.array,
                          diff_method='abs',
                          threshold=3,
                          required_yes=1,
                          veto_limit=0,
                          index_of_interest=-1)

    # test cell of interest is set to 'False' with veto
    assert result_3[x_interest, y_interest] == False


def test_uncertainty(sample_punchdata: NDCube):
    # create an uncertainty array of 0's
    sample_punchdata.uncertainty.array[...] = 0

    # choose a pixel of interest
    x_test_px=210
    y_test_px=355
    index_of_interest=-1

   # initial test to see pixel of interest is normal
    result_0 = run_zspike(sample_punchdata.data,
                          sample_punchdata.uncertainty.array,
                          diff_method='abs',
                          threshold=3,
                          required_yes=1,
                          veto_limit=1,
                          index_of_interest=index_of_interest)

    # test cell of interest is set to 'True'
    assert result_0[y_test_px, x_test_px] == False

    # set pixel of interest to high value
    sample_punchdata.data[index_of_interest, y_test_px, x_test_px] = 1000

    result_1 = run_zspike(sample_punchdata.data,
                          sample_punchdata.uncertainty.array,
                          diff_method='abs',
                          threshold=3,
                          required_yes=1,
                          veto_limit=1,
                          index_of_interest=index_of_interest)

    # test cell of interest is set to 'True'
    assert result_1[y_test_px, x_test_px] == True

    # make bad pixels adjacent to cell of interest also high
    # set pixel of interest to high value
    sample_punchdata.data[:, y_test_px, x_test_px]=1000

    result_2 = run_zspike(sample_punchdata.data,
                          sample_punchdata.uncertainty.array,
                          diff_method='abs',
                          threshold=3,
                          required_yes=1,
                          veto_limit=1,
                          index_of_interest=index_of_interest)

    # test cell of interest is set to 'False'
    assert result_2[y_test_px, x_test_px] == False

    # set surrounding values to uncertain

    sample_punchdata.uncertainty.array[index_of_interest, y_test_px, x_test_px] = np.inf
    result_3 = run_zspike(sample_punchdata.data,
                          sample_punchdata.uncertainty.array,
                          diff_method='abs',
                          threshold=3,
                          required_yes=1,
                          veto_limit=1,
                          index_of_interest=index_of_interest)
    # test cell of interest is set to 'True' due to uncertainty flag set on the surrounding pixels
    assert result_3[y_test_px, x_test_px] == True




def test_threshold_abs(sample_punchdata: NDCube):
    # create an uncertainty array of 0's
    sample_punchdata.uncertainty.array[:, :, :] = 0

    # choose a pixel of interest
    x_test_px=210
    y_test_px=355
    index_of_interest=-1

   # initial test to see pixel of interest is normal
    result_0 = run_zspike(sample_punchdata.data,
                          sample_punchdata.uncertainty.array,
                          diff_method='abs',
                          threshold=3,
                          required_yes=1,
                          veto_limit=1,
                          index_of_interest=index_of_interest)


    assert result_0[y_test_px, x_test_px] == False

    # set pixel of interest to high value
    sample_punchdata.data[index_of_interest, y_test_px, x_test_px]=100

    result_1 = run_zspike(sample_punchdata.data,
                          sample_punchdata.uncertainty.array,
                          diff_method='abs',
                          threshold=3,
                          required_yes=1,
                          veto_limit=1,
                          index_of_interest=index_of_interest)

    assert result_1[y_test_px, x_test_px] == True

    # make bad pixel threshold high

    result_2 = run_zspike(sample_punchdata.data,
                          sample_punchdata.uncertainty.array,
                          diff_method='abs',
                          threshold=300,
                          required_yes=1,
                          veto_limit=1,
                          index_of_interest=index_of_interest)

    # test cell of interest is set to 'False'
    assert result_2[y_test_px, x_test_px] == False


def test_threshold_sigma(sample_punchdata: NDCube):
    # create an uncertainty array of 0's
    sample_punchdata.uncertainty.array[:, :, :] = 0
    sample_punchdata.data[:, :, :] = 0

    # choose a pixel of interest
    x_test_px=210
    y_test_px=355
    index_of_interest=-1

    # initial test to see pixel of interest is normal
    result_0 = run_zspike(sample_punchdata.data,
                          sample_punchdata.uncertainty.array,
                          diff_method='sigma',
                          threshold=3,
                          required_yes=1,
                          veto_limit=1,
                          index_of_interest=index_of_interest)


    assert result_0[y_test_px, x_test_px] == False

    # set pixel of interest to high value
    sample_punchdata.data[index_of_interest, y_test_px, x_test_px]=100


    result_1 = run_zspike(sample_punchdata.data,
                          sample_punchdata.uncertainty.array,
                          diff_method='sigma',
                          threshold=1,
                          required_yes=1,
                          veto_limit=1,
                          index_of_interest=index_of_interest)

    assert result_1[y_test_px, x_test_px] == True

    # make bad pixel threshold high

    result_2 = run_zspike(sample_punchdata.data,
                          sample_punchdata.uncertainty.array,
                          diff_method='abs',
                          threshold=300,
                          required_yes=1,
                          veto_limit=1,
                          index_of_interest=index_of_interest)

    # test cell of interest is set to 'False'
    assert result_2[y_test_px, x_test_px] == False



def test_required_yes_abs(sample_punchdata: NDCube):
    # create an uncertainty array of 0's
    sample_punchdata.uncertainty.array[:, :, :] = 0

    # choose a pixel of interest
    x_test_px=210
    y_test_px=355
    index_of_interest=-1


    # set pixel of interest to high value

    result = run_zspike(sample_punchdata.data,
                        sample_punchdata.uncertainty.array,
                        diff_method='abs',
                        threshold=1,
                        required_yes=1,
                        veto_limit=1,
                        index_of_interest=index_of_interest)

    assert result[y_test_px, x_test_px] == False

    #sample_punchdata.data[0:3, y_test_px, x_test_px]=100
    sample_punchdata.data[index_of_interest, y_test_px, x_test_px]=100

    # set pixel of interest to high value

    result_1 = run_zspike(sample_punchdata.data,
                          sample_punchdata.uncertainty.array,
                          diff_method='abs',
                          threshold=1,
                          required_yes=1,
                          veto_limit=1,
                          index_of_interest=index_of_interest)

    assert result_1[y_test_px, x_test_px] == True

    # change the number adjacent elements that have high values, this
    # reduces the number of available yes voters, making the cell of
    # interest 'false'
    sample_punchdata.data[0:1, y_test_px, x_test_px]=100
    sample_punchdata.data[index_of_interest, y_test_px, x_test_px]=100

    # set pixel of interest to high value

    result_2 = run_zspike(sample_punchdata.data,
                          sample_punchdata.uncertainty.array,
                          diff_method='abs',
                          threshold=1,
                          required_yes=1,
                          veto_limit=1,
                          index_of_interest=index_of_interest)

    assert result_2[y_test_px, x_test_px] == True

    # change the number adjacent elements that have high values
    sample_punchdata.data[0:1, y_test_px, x_test_px]=100
    sample_punchdata.data[index_of_interest, y_test_px, x_test_px]=100

    # set pixel of interest to high value

    result_3 = run_zspike(sample_punchdata.data,
                          sample_punchdata.uncertainty.array,
                          diff_method='abs',
                          threshold=1,
                          required_yes=3,
                          veto_limit=1,
                          index_of_interest=index_of_interest)

    assert result_3[y_test_px, x_test_px] == False


def test_required_yes_sigma(sample_punchdata: NDCube):
    # create an uncertainty and data array of 0's
    sample_punchdata.uncertainty.array[:, :, :] = 0
    sample_punchdata.data[:, :, :] = 0
    # choose a pixel of interest
    x_test_px=210
    y_test_px=355
    index_of_interest=-1


    result = run_zspike(sample_punchdata.data,
                        sample_punchdata.uncertainty.array,
                        diff_method='sigma',
                        threshold=1,
                        required_yes=1,
                        veto_limit=1,
                        index_of_interest=index_of_interest)

    assert result[y_test_px, x_test_px] == False

    #sample_punchdata.data[0:3, y_test_px, x_test_px]=100
    sample_punchdata.data[index_of_interest, y_test_px, x_test_px]=10

    # set pixel of interest to high value

    result_1 = run_zspike(sample_punchdata.data,
                          sample_punchdata.uncertainty.array,
                          diff_method='sigma',
                          threshold=1,
                          required_yes=1,
                          veto_limit=1,
                          index_of_interest=index_of_interest)

    assert result_1[y_test_px, x_test_px] == True

    # change the number adjacent elements that have high values, this
    # reduces the number of available yes voters, making the cell of
    # interest 'false'
    sample_punchdata.data[0:1, y_test_px, x_test_px]=10
    sample_punchdata.data[index_of_interest, y_test_px, x_test_px]=10

    # set pixel of interest to high value

    result_2 = run_zspike(sample_punchdata.data,
                          sample_punchdata.uncertainty.array,
                          diff_method='sigma',
                          threshold=1,
                          required_yes=1,
                          veto_limit=1,
                          index_of_interest=index_of_interest)

    assert result_2[y_test_px, x_test_px] == True

    # change the number adjacent elements that have high values
    sample_punchdata.data[0:1, y_test_px, x_test_px]=10
    sample_punchdata.data[index_of_interest, y_test_px, x_test_px]=10

    # set pixel of interest to high value

    result_3 = run_zspike(sample_punchdata.data,
                          sample_punchdata.uncertainty.array,
                          diff_method='sigma',
                          threshold=1,
                          required_yes=3,
                          veto_limit=1,
                          index_of_interest=index_of_interest)

    assert result_3[y_test_px, x_test_px] == False

def test_dilation_abs(sample_punchdata: NDCube):
        # create an uncertainty and data array of 0's
    sample_punchdata.uncertainty.array[:, :, :] = 0
    #sample_punchdata.data[:, :, :] = 0
    # choose a pixel of interest
    x_test_px=210
    y_test_px=355
    index_of_interest=-1


    result = run_zspike(sample_punchdata.data,
                        sample_punchdata.uncertainty.array,
                        diff_method='abs',
                        threshold=1,
                        required_yes=1,
                        veto_limit=1,
                        index_of_interest=index_of_interest)

    assert result[y_test_px, x_test_px] == False

    #sample_punchdata.data[0:3, y_test_px, x_test_px]=100
    sample_punchdata.data[index_of_interest, y_test_px, x_test_px]=10

    # set pixel of interest to high value

    result_1 = run_zspike(sample_punchdata.data,
                          sample_punchdata.uncertainty.array,
                          diff_method='sigma',
                          threshold=1,
                          required_yes=1,
                          veto_limit=1,
                          index_of_interest=index_of_interest)

    assert result_1[y_test_px, x_test_px] == True

    # change the number adjacent elements that have high values, this
    # reduces the number of available yes voters, making the cell of
    # interest 'false'
    sample_punchdata.data[0:1, y_test_px, x_test_px]=100
    sample_punchdata.data[index_of_interest, y_test_px, x_test_px]=100

    # set pixel of interest to high value

    result_2 = run_zspike(sample_punchdata.data,
                          sample_punchdata.uncertainty.array,
                          diff_method='abs',
                          threshold=1,
                          required_yes=1,
                          veto_limit=1,
                          index_of_interest=index_of_interest)


    assert result_2[y_test_px, x_test_px] == True
    # with no dilation a an adjacent pixel is false
    assert result_2[y_test_px+1, x_test_px] == False

    result_3 = run_zspike(sample_punchdata.data,
                          sample_punchdata.uncertainty.array,
                          diff_method='sigma',
                          threshold=1,
                          required_yes=1,
                          veto_limit=1,
                          index_of_interest=index_of_interest,
                          dilation=10)


    assert result_3[y_test_px, x_test_px] == True
    # with dilation an adjacent pixel is true
    assert result_3[y_test_px+1, x_test_px] == True



    # change the number adjacent elements that have high values
    sample_punchdata.data[0:1, y_test_px, x_test_px]=10
    sample_punchdata.data[index_of_interest, y_test_px, x_test_px]=10



def test_dilation_sigma(sample_punchdata: NDCube):
        # create an uncertainty and data array of 0's
    sample_punchdata.uncertainty.array[:, :, :] = 0
    sample_punchdata.data[:, :, :] = 0
    # choose a pixel of interest
    x_test_px=210
    y_test_px=355
    index_of_interest=-1


    result = run_zspike(sample_punchdata.data,
                        sample_punchdata.uncertainty.array,
                        diff_method='sigma',
                        threshold=1,
                        required_yes=1,
                        veto_limit=1,
                        index_of_interest=index_of_interest)

    assert result[y_test_px, x_test_px] == False

    #sample_punchdata.data[0:3, y_test_px, x_test_px]=100
    sample_punchdata.data[index_of_interest, y_test_px, x_test_px]=10

    # set pixel of interest to high value

    result_1 = run_zspike(sample_punchdata.data,
                          sample_punchdata.uncertainty.array,
                          diff_method='sigma',
                          threshold=1,
                          required_yes=1,
                          veto_limit=1,
                          index_of_interest=index_of_interest)

    assert result_1[y_test_px, x_test_px] == True

    # change the number adjacent elements that have high values, this
    # reduces the number of available yes voters, making the cell of
    # interest 'false'
    sample_punchdata.data[0:1, y_test_px, x_test_px]=10
    sample_punchdata.data[index_of_interest, y_test_px, x_test_px]=10

    # set pixel of interest to high value

    result_2 = run_zspike(sample_punchdata.data,
                          sample_punchdata.uncertainty.array,
                          diff_method='sigma',
                          threshold=1,
                          required_yes=1,
                          veto_limit=1,
                          index_of_interest=index_of_interest)


    assert result_2[y_test_px, x_test_px] == True
    # with no dilation a an adjacent pixel is false
    assert result_2[y_test_px+1, x_test_px] == False

    result_3 = run_zspike(sample_punchdata.data,
                          sample_punchdata.uncertainty.array,
                          diff_method='sigma',
                          threshold=1,
                          required_yes=1,
                          veto_limit=1,
                          index_of_interest=index_of_interest,
                          dilation=10)


    assert result_3[y_test_px, x_test_px] == True
    # with dilation an adjacent pixel is true
    assert result_3[y_test_px+1, x_test_px] == True



    # change the number adjacent elements that have high values
    sample_punchdata.data[0:1, y_test_px, x_test_px]=10
    sample_punchdata.data[index_of_interest, y_test_px, x_test_px]=10
