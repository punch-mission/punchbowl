"""Creates test data with the appropriate metadata for punchbowl"""
import os
from datetime import datetime, timedelta

import numpy as np
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS

from punchbowl.data import NormalizedMetadata, PUNCHData
from punchbowl.level1.quartic_fit import create_constant_quartic_coefficients


def create_f_corona_test_data(path="../punchbowl/level3/tests/data/"):
    meta = NormalizedMetadata.load_template("CFM", "3")
    meta["DATE-OBS"] = str(datetime.now())
    wcs = WCS(naxis=2)
    for i in range(10):
        data = np.ones((10, 10)) * i
        obj = PUNCHData(data, wcs, meta)
        file_path = os.path.join(path, f"test_{i}.fits")
        obj.write(file_path, overwrite=True, skip_wcs_conversion=True)


def create_punchdata_test_data(path="../punchbowl/tests/"):
    meta = NormalizedMetadata.load_template("CFM", "3")
    meta["DATE-OBS"] = str(datetime.now())
    wcs = WCS(naxis=2)
    data = np.ones((10, 10))
    obj = PUNCHData(data, wcs, meta)
    file_path = os.path.join(path, "test_data.fits")
    obj.write(file_path, overwrite=True, skip_wcs_conversion=True)


def create_header_validation_test_data(path="../punchbowl/tests/"):
    m = NormalizedMetadata.load_template("PTM", "3")
    m['DATE-OBS'] = datetime.utcnow().isoformat()
    m['DATE-BEG'] = datetime.utcnow().isoformat()
    m['DATE-AVG'] = (datetime.utcnow() + timedelta(minutes=2)).isoformat()
    m['DATE-END'] = (datetime.utcnow() + timedelta(minutes=4)).isoformat()
    m['DATE'] = (datetime.utcnow() + timedelta(hours=1)).isoformat()
    h = m.to_fits_header()

    data = np.ones((2, 4096, 4096), dtype=np.float32)
    uncertainty = StdDevUncertainty(np.sqrt(np.abs(data)).astype(np.int8))

    d = PUNCHData(data=data, uncertainty=uncertainty, wcs=WCS(h), meta=m)
    file_path = os.path.join(path, "test_header_validation.fits")
    d.write(file_path, overwrite=True)


def create_quartic_coefficients_test_data(path="../punchbowl/level1/tests/data/"):
    meta = NormalizedMetadata.load_template("CF1", "1")
    meta['DATE-OBS'] = str(datetime.now())
    wcs = WCS(naxis=3)
    data = create_constant_quartic_coefficients((10, 10))
    obj = PUNCHData(data, wcs, meta)
    file_path = os.path.join(path, "test_quartic_coeffs.fits")
    obj.write(file_path, overwrite=True, skip_wcs_conversion=True)

def create_vignetting_test_data(path="../punchbowl/level1/tests/data/"):
    meta = NormalizedMetadata.load_template("GR1", "1")
    meta['DATE-OBS'] = str(datetime.now())
    wcs = WCS(naxis=2)
    data = np.load('data/sample_vignetting.npy')
    obj = PUNCHData(data, wcs, meta)
    file_path = os.path.join(path, obj.filename_base + '.fits')
    obj.write(file_path, overwrite=True, skip_wcs_conversion=True)


if __name__ == "__main__":
    create_header_validation_test_data()
    create_f_corona_test_data()
    create_punchdata_test_data()
    create_quartic_coefficients_test_data()
    create_vignetting_test_data()
