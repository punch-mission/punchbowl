import numpy as np
from astropy.wcs import WCS
from astropy.nddata import StdDevUncertainty
from ndcube import NDCube
from pytest import fixture

# punchbowl imports
from punchbowl.data import PUNCHData


# Some test inputs
@fixture
def sample_wcs() -> WCS:
    """
    Generate a sample WCS for testing
    """

    def _sample_wcs(naxis=2, crpix=(0, 0), crval=(0, 0), cdelt=(1, 1),
                    ctype=("HPLN-ARC", "HPLT-ARC")):
        generated_wcs = WCS(naxis=naxis)

        generated_wcs.wcs.crpix = crpix
        generated_wcs.wcs.crval = crval
        generated_wcs.wcs.cdelt = cdelt
        generated_wcs.wcs.ctype = ctype

        return generated_wcs

    return _sample_wcs


@fixture
def sample_data(shape: tuple = (20, 20)) -> np.ndarray:
    """
    Generate some random data for testing
    """

    return np.random.random(shape)


@fixture
def sample_ndcube(sample_data):
    """
    Generate a sample ndcube object for testing
    """

    data = sample_data
    uncertainty = StdDevUncertainty(np.sqrt(np.abs(sample_data)))
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.1, 0.1
    wcs.wcs.crpix = 0, 0
    wcs.wcs.crval = 1, 1
    wcs.wcs.cname = "HPC lon", "HPC lat"
    nd_obj = NDCube(data=data, uncertainty=uncertainty, wcs=wcs)
    return nd_obj


@fixture
def sample_punchdata(sample_ndcube):
    """
    Generate a sample PUNCHData object for testing
    """

    sample_ndc = sample_ndcube
    return PUNCHData(sample_ndc)


@fixture
def sample_punchdata_list(sample_punchdata):
    """
    Generate a list of sample PUNCHData objects for testing
    """

    sample_pd1 = sample_punchdata
    sample_pd2 = sample_punchdata
    return list([sample_pd1, sample_pd2])
