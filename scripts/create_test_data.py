"""Creates test data with the appropriate metadata for punchbowl"""
import os
from datetime import datetime

import numpy as np
from astropy.wcs import WCS

from punchbowl.data import NormalizedMetadata, PUNCHData


def create_f_corona_test_data(path="../punchbowl/level3/tests/data/"):
    meta = NormalizedMetadata.load_template("CFM", "3")
    meta["DATE-OBS"] = str(datetime.now())
    wcs = WCS(naxis=2)
    for i in range(10):
        data = np.ones((10, 10)) * i
        obj = PUNCHData(data, wcs, meta)
        file_path = os.path.join(path, f"test_{i}.fits")
        obj.write(file_path, overwrite=True)


def create_punchdata_test_data(path="../punchbowl/tests/"):
    meta = NormalizedMetadata.load_template("CFM", "3")
    meta["DATE-OBS"] = str(datetime.now())
    wcs = WCS(naxis=2)
    data = np.ones((10, 10))
    obj = PUNCHData(data, wcs, meta)
    file_path = os.path.join(path, "test_data.fits")
    obj.write(file_path, overwrite=True)


if __name__ == "__main__":
    create_f_corona_test_data()
    create_punchdata_test_data()
