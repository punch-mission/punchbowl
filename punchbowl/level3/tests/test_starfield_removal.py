import unittest
import os
import pathlib
from glob import glob
from pkg_resources import iter_entry_points

# Third party imports
import numpy as np
import pytest
from astropy.wcs import WCS
from remove_starfield import Starfield

# punchbowl imports
from punchbowl.data import NormalizedMetadata, PUNCHData
from punchbowl.level3.starfield_remove import (
    generate_starfield_background,
    subtract_starfield_background,
    subtract_starfield_background_task)


@pytest.fixture()
def one_data(shape: tuple = (128, 128)) -> PUNCHData:
    """
    Generate some random data for testing
    """
    data = np.ones(shape)

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLN-AZP", "HPLT-AZP"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.02, 0.02
    wcs.wcs.crpix = 1024, 1024
    wcs.wcs.crval = 0, 24.75

    meta = NormalizedMetadata(
        {"TYPECODE": "CS", "OBSCODE": "M", "LEVEL": "3", "OBSRVTRY": "0", "DATE-OBS": "2024-04-08T18:40:00"})
    return PUNCHData(data=data, wcs=wcs, meta=meta)


@pytest.fixture()
def zero_starfield_data(shape: tuple = (256, 256)) -> Starfield:
    """
    Generate some random data for testing
    """
    starfield = np.zeros(shape)

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLN-AZP", "HPLT-AZP"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.02, 0.02
    wcs.wcs.crpix = 1024, 1024
    wcs.wcs.crval = 0, 24.75

    # meta = NormalizedMetadata(
    #     {"TYPECODE": "CS", "OBSCODE": "M", "LEVEL": "3", "OBSRVTRY": "0", "DATE-OBS": "2024-04-08T18:40:00"})
    return Starfield(starfield=starfield, wcs=wcs)


def test_basic_subtraction(one_data: PUNCHData, zero_starfield_data: Starfield) -> None:
    """

    """
    subtraction_starfield = subtract_starfield_background(one_data, zero_starfield_data)
    subtraction_punchdata = PUNCHData(data=subtraction_starfield.data, wcs=subtraction_starfield.wcs, meta=subtraction_starfield.meta)
    assert isinstance(subtraction_punchdata, PUNCHData)
    assert np.allclose(subtraction_punchdata.data, 1)
