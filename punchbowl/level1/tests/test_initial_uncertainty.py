import pathlib
from datetime import datetime

import numpy as np
import pytest
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube

from punchbowl.data import NormalizedMetadata
from punchbowl.level1.initial_uncertainty import flag_saturated_pixels

THIS_DIRECTORY = pathlib.Path(__file__).parent.resolve()


@pytest.fixture()
def sample_punchdata(shape: tuple = (2048, 2048)) -> NDCube:
    """
    Generate a sample PUNCH data object for testing
    """

    data = np.random.random(shape)
    uncertainty = StdDevUncertainty(np.zeros_like(data))

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLN-AZP", "HPLT-AZP"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.02, 0.02
    wcs.wcs.crpix = 1024, 1024
    wcs.wcs.crval = 0, 24.75
    wcs.array_shape = shape

    meta = NormalizedMetadata.load_template("CR3", "0")
    meta['DATE-OBS'] = str(datetime(2024, 2, 22, 16, 0, 1))

    return NDCube(data=data, uncertainty=uncertainty, wcs=wcs, meta=meta)


def test_flag_saturated_pixels(sample_punchdata: NDCube) -> None:
    cube = sample_punchdata

    cube.meta["COMPBITS"] = 10
    cube.meta["DSATVAL"] = 2**cube.meta["COMPBITS"].value-1

    n_saturated_pixels = 20
    x_coords = np.fix(np.random.random(n_saturated_pixels) * cube.data.shape[0]).astype(int)
    y_coords = np.fix(np.random.random(n_saturated_pixels) * cube.data.shape[1]).astype(int)

    cube.data[x_coords, y_coords] = cube.meta["DSATVAL"].value
    saturated_pixels = cube.data >= cube.meta["DSATVAL"].value

    cube = flag_saturated_pixels(cube, saturated_pixels)

    assert np.all(np.isposinf(cube.uncertainty.array[x_coords, y_coords]))
