from punchbowl.data import PUNCHData

import numpy as np
from ndcube import NDCube
from prefect import task, get_run_logger
from astropy.wcs import WCS
import astropy.units as u
from typing import Dict, Any


@task
def decode_ccsds_packets(path: str):
    pass


def create_fake_ndcube():
    data = np.random.rand(2048, 2048)
    uncertainty = np.zeros((2048, 2048))
    meta = {}

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLT-TAN", "HPLN-TAN"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.5, 0.5
    wcs.wcs.crpix = 1024, 1024
    wcs.wcs.crval = 0, 0
    wcs.wcs.cname = "HPC lat", "HPC lon"
    nd_obj = NDCube(data=data, wcs=wcs, uncertainty=uncertainty, meta=meta, unit=u.ct)
    return nd_obj


@task
def create_level0_from_packets(packet_contents):
    cube = create_fake_ndcube()
    level0 = PUNCHData(cube)
    return level0


@task
def write_level0(level0: PUNCHData, path: str) -> Dict[str, Any]:
    return level0.write(path)
