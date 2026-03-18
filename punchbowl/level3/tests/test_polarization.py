import os
import pytest
import numpy as np
import astropy.units as u
from ndcube import NDCube
from punchbowl.data.punch_io import load_ndcube_from_fits
from punchbowl.data.tests.test_punch_io import sample_ndcube
from punchbowl.level3.polarization import convert_polarization
from solpolpy.alpha import radial_north

tbarray = np.ones((512, 512), dtype=np.float32)
pbarray = 0.5 * tbarray
pbparray = 0 * tbarray

TESTDATA_DIR = os.path.dirname(__file__)
TEST_FILE = TESTDATA_DIR + '/data/downsampled_L2_PTM.fits'

@pytest.fixture
def sample_data_triplet(sample_ndcube):
    """
    Generate a list of sample PUNCH data objects for testing polarization resolving
    """
    data_cube = load_ndcube_from_fits(TEST_FILE, include_provenance=False)
    shape = data_cube.data[0].shape
    alph = radial_north(shape)

    polar_angles = [-60, 0, 60] * u.degree
    Bm = 0.5 * (tbarray - pbarray * np.cos(2 * (polar_angles[0] - alph)))
    Bz = 0.5 * (tbarray - pbarray * np.cos(2 * (polar_angles[1] - alph)))
    Bp = 0.5 * (tbarray - pbarray * np.cos(2 * (polar_angles[2] - alph)))

    data_mzp = np.stack((Bm, Bz, Bp))

    return NDCube(data=data_mzp, wcs=data_cube.wcs, meta=data_cube.meta)


def test_convert_polarization(sample_data_triplet):
    output = convert_polarization(sample_data_triplet)

    assert isinstance(output, NDCube)
    assert output.data[0].shape == tbarray.shape
    assert np.allclose(output.data[0], tbarray)
    assert np.allclose(output.data[1], pbarray)
    assert np.allclose(output.data[2], pbparray)


pbparray_nz = 0.5 * tbarray
@pytest.fixture
def sample_data_triplet_pbp_nonzero(sample_ndcube):
    """
    Generate a list of sample PUNCH data objects for testing polarization resolving
    """
    data_cube = load_ndcube_from_fits(TEST_FILE, include_provenance=False)
    shape = data_cube.data[0].shape
    alph = radial_north(shape)

    polar_angles = [-60, 0, 60] * u.degree
    Bm = (1 / 2) * (tbarray - pbarray * np.cos(2 * (polar_angles[0] - alph)) -
                    pbparray_nz * np.sin(2 * (polar_angles[0] - alph)))
    Bz = (1 / 2) * (tbarray - pbarray * np.cos(2 * (polar_angles[1] - alph)) -
                    pbparray_nz * np.sin(2 * (polar_angles[1] - alph)))
    Bp = (1 / 2) * (tbarray - pbarray * np.cos(2 * (polar_angles[2] - alph)) -
                    pbparray_nz * np.sin(2 * (polar_angles[2] - alph)))

    data_mzp = np.stack((Bm, Bz, Bp))

    return NDCube(data=data_mzp, wcs=data_cube.wcs, meta=data_cube.meta)


def test_convert_polarization_pbp_nonzero(sample_data_triplet_pbp_nonzero):
    output = convert_polarization(sample_data_triplet_pbp_nonzero)

    assert isinstance(output, NDCube)
    assert output.data[0].shape == tbarray.shape
    assert np.allclose(output.data[0], tbarray)
    assert np.allclose(output.data[1], pbarray)
    assert np.allclose(output.data[2], pbparray_nz)


@pytest.fixture
def sample_data_triplet_nopbp_equation(sample_ndcube):
    """
    Generate a list of sample PUNCH data objects for testing polarization resolving
    """
    data_cube = load_ndcube_from_fits(TEST_FILE, include_provenance=False)
    shape = data_cube.data[0].shape
    alph = radial_north(shape)

    polar_angles = [-60, 0, 60] * u.degree
    Bm = (1 / 2) * (tbarray - pbarray * np.cos(2 * (polar_angles[0] - alph)))
    Bz = (1 / 2) * (tbarray - pbarray * np.cos(2 * (polar_angles[1] - alph)))
    Bp = (1 / 2) * (tbarray - pbarray * np.cos(2 * (polar_angles[2] - alph)))

    data_mzp = np.stack((Bm, Bz, Bp))

    return NDCube(data=data_mzp, wcs=data_cube.wcs, meta=data_cube.meta)


def test_convert_polarization_nopbp_equation(sample_data_triplet_nopbp_equation):
    output = convert_polarization(sample_data_triplet_nopbp_equation)
    assert np.allclose(output.data[2], pbparray_nz) == False