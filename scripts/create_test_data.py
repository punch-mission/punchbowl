"""Creates test data with the appropriate metadata for punchbowl"""
import os
from datetime import datetime, timedelta

import numpy as np
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube

from punchbowl.data import NormalizedMetadata, get_base_file_name, write_ndcube_to_fits
from punchbowl.level1.quartic_fit import create_constant_quartic_coefficients


def create_f_corona_test_data(path="../punchbowl/level3/tests/data/"):
    meta = NormalizedMetadata.load_template("CFM", "3")
    meta["DATE-OBS"] = str(datetime.now())
    wcs = WCS(naxis=2)
    for i in range(10):
        data = np.ones((3, 10, 10)) * i
        obj = NDCube(data=data, wcs=wcs, meta=meta)
        file_path = os.path.join(path, f"test_{i}.fits")
        write_ndcube_to_fits(obj, file_path, overwrite=True)


# def create_punchdata_test_data(path="../punchbowl/tests/"):
#     meta = NormalizedMetadata.load_template("CFM", "3")
#     meta["DATE-OBS"] = str(datetime.now())
#     wcs = WCS(naxis=2)
#     data = np.ones((10, 10))
#     obj = NDCube(data=data, wcs=wcs, meta=meta)
#     file_path = os.path.join(path, "test_data.fits")
#     write_ndcube_to_fits(obj, file_path, overwrite=True)


# def create_header_validation_test_data(path="../punchbowl/tests/"):
#     wcs = WCS(naxis=2)
#     m = NormalizedMetadata.load_template("PTM", "3")
#     m['DATE-OBS'] = datetime.utcnow().isoformat()
#     m['DATE-BEG'] = datetime.utcnow().isoformat()
#     m['DATE-AVG'] = (datetime.utcnow() + timedelta(minutes=2)).isoformat()
#     m['DATE-END'] = (datetime.utcnow() + timedelta(minutes=4)).isoformat()
#     m['DATE'] = (datetime.utcnow() + timedelta(hours=1)).isoformat()
#     h = m.to_fits_header(wcs=wcs)
#
#     data = np.ones((2, 4096, 4096), dtype=np.float32)
#     uncertainty = StdDevUncertainty(np.sqrt(np.abs(data)).astype(np.float32))
#
#     d = NDCube(data=data, uncertainty=uncertainty, wcs=WCS(h), meta=m)
#     file_path = os.path.join(path, "test_header_validation.fits")
#     write_ndcube_to_fits(d, file_path, overwrite=True)


def create_quartic_coefficients_test_data(path="../punchbowl/level1/tests/data/"):
    meta = NormalizedMetadata.load_template("FQ1", "1")
    meta['DATE-OBS'] = str(datetime.now())
    wcs = WCS(naxis=3)
    data = create_constant_quartic_coefficients((10, 10))
    obj = NDCube(data=data, wcs=wcs, meta=meta)
    file_path = os.path.join(path, "test_quartic_coeffs.fits")
    write_ndcube_to_fits(obj, file_path, overwrite=True)


def create_vignetting_test_data(path="../punchbowl/level1/tests/data/"):
    meta = NormalizedMetadata.load_template("GR1", "1")
    meta['DATE-OBS'] = str(datetime(2024,2, 22, 16,34, 25))
    wcs = WCS(naxis=2)
    data = np.random.random((10, 10))
    obj = NDCube(data=data, wcs=wcs, meta=meta)
    file_path = os.path.join(path, get_base_file_name(obj) + '.fits')
    write_ndcube_to_fits(obj, file_path, overwrite=True)

def create_stray_light_test_data(path="../punchbowl/level1/tests/data/"):
    meta = NormalizedMetadata.load_template("SM1", "1")
    meta['DATE-OBS'] = str(datetime(2024,2, 22, 16,34, 25))
    wcs = WCS(naxis=2)
    data = np.random.random((10, 10))
    obj = NDCube(data=data, wcs=wcs, meta=meta)
    file_path = os.path.join(path, get_base_file_name(obj) + '.fits')
    write_ndcube_to_fits(obj, file_path, overwrite=True)

if __name__ == "__main__":
    # create_header_validation_test_data()
    create_f_corona_test_data()
    # create_punchdata_test_data()
    create_quartic_coefficients_test_data()
    create_stray_light_test_data()
    create_vignetting_test_data()
