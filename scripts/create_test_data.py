"""Creates test data with the appropriate metadata for punchbowl"""
import os
from io import BytesIO
from datetime import UTC, datetime

import astropy.nddata
import numpy as np
import requests
from astropy.io import fits
from astropy.wcs import WCS

from punchbowl.data import NormalizedMetadata, get_base_file_name, write_ndcube_to_fits
from punchbowl.data.punchcube import PUNCHCube
from punchbowl.level1.quartic_fit import create_constant_quartic_coefficients


def create_L2_PTM_test_file(path="../punchbowl/level3/tests/data/"):
    src_file = "https://data.boulder.swri.edu/svankooten/PUNCH_L2_PTM_20241106032023_v1.fits"
    r = requests.get(src_file)
    with fits.open(BytesIO(r.content)) as hdul:
        for hdu in hdul[1:]:
            hdu.data = astropy.nddata.block_reduce(hdu.data, (1, 8, 8), np.nanmean)
            for key in ' ', 'A':
                wcs = WCS(hdu.header, key=key)
                w = wcs[:, ::8, ::8]
                hdu.header.update(w.to_header(key=key))
        file_path = os.path.join(path, "downsampled_L2_PTM.fits")
        hdul.writeto(file_path, overwrite=True)


def create_f_corona_test_data(path="../punchbowl/level3/tests/data/"):
    meta = NormalizedMetadata.load_template("CFM", "3")
    meta["DATE-OBS"] = str(datetime.now(UTC))
    wcs = WCS(naxis=2)
    for i in range(10):
        data = np.ones((3, 10, 10)) * i
        obj = PUNCHCube(data=data, wcs=wcs, meta=meta)
        file_path = os.path.join(path, f"test_{i}.fits")
        write_ndcube_to_fits(obj, file_path, overwrite=True)


def create_quartic_coefficients_test_data(path="../punchbowl/level1/tests/data/"):
    meta = NormalizedMetadata.load_template("FR1", "1")
    meta['DATE-OBS'] = str(datetime.now(UTC))
    wcs = WCS(naxis=3)
    data = create_constant_quartic_coefficients((10, 10))
    obj = PUNCHCube(data=data, wcs=wcs, meta=meta)
    file_path = os.path.join(path, "test_quartic_coeffs.fits")
    write_ndcube_to_fits(obj, file_path, overwrite=True)


def create_vignetting_test_data(path="../punchbowl/level1/tests/data/"):
    meta = NormalizedMetadata.load_template("GR1", "1")
    meta['DATE-OBS'] = str(datetime(2024,2, 22, 16,34, 25))
    wcs = WCS(naxis=2)
    data = np.random.random((10, 10))
    obj = PUNCHCube(data=data, wcs=wcs, meta=meta)
    file_path = os.path.join(path, get_base_file_name(obj) + '.fits')
    write_ndcube_to_fits(obj, file_path, overwrite=True)

def create_stray_light_test_data(path="../punchbowl/level1/tests/data/"):
    meta = NormalizedMetadata.load_template("SM1", "1")
    meta['DATE-OBS'] = str(datetime(2024,2, 22, 16,34, 25))
    wcs = WCS(naxis=2)
    data = np.random.random((10, 10))
    obj = PUNCHCube(data=data, wcs=wcs, meta=meta)
    file_path = os.path.join(path, get_base_file_name(obj) + '.fits')
    write_ndcube_to_fits(obj, file_path, overwrite=True)

if __name__ == "__main__":
    # create_header_validation_test_data()
    create_f_corona_test_data()
    # create_punchdata_test_data()
    create_quartic_coefficients_test_data()
    create_stray_light_test_data()
    create_vignetting_test_data()
