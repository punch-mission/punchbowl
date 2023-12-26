# from typing import Any, Dict
#
# import astropy.units as u
# import numpy as np
# from astropy.wcs import WCS
# from ndcube import NDCube
# from prefect import task
#
# from punchbowl.data import PUNCHData
#
#
# @task
# def decode_ccsds_packets(path: str):
#     pass
#
#
# def create_fake_ndcube():
#     data = np.random.rand(2048, 2048)
#     uncertainty = np.zeros((2048, 2048))
#     meta = {}
#
#     wcs = WCS(naxis=2)
#     wcs.wcs.ctype = "HPLT-TAN", "HPLN-TAN"
#     wcs.wcs.cunit = "deg", "deg"
#     wcs.wcs.cdelt = 0.5, 0.5
#     wcs.wcs.crpix = 1024, 1024
#     wcs.wcs.crval = 0, 0
#     wcs.wcs.cname = "HPC lat", "HPC lon"
#     nd_obj = NDCube(data=data, wcs=wcs, uncertainty=uncertainty, meta=meta, unit=u.ct)
#     return nd_obj
#
#
# @task
# def create_level0_from_packets(packet_contents):
#     cube = create_fake_ndcube()
#     return PUNCHData(cube)
#
#
# @task
# def write_level0(level0: PUNCHData, path: str) -> Dict[str, Any]:
#     return level0.write(path)
