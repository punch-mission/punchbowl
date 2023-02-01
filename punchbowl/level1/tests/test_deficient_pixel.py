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
from punchbowl.data import PUNCHData,NormalizedMetadata
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
