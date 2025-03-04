import os
from datetime import datetime

import numpy as np
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from astropy.wcs.utils import add_stokes_axis_to_wcs
from ndcube import NDCube

from punchbowl.data import NormalizedMetadata, get_base_file_name, write_ndcube_to_fits
from punchbowl.data.meta import load_level_spec

LEVELS = ["0", "1", "2", "3", "L", "Q"]


def sample_ndcube(shape, code="PM1", level="0"):
    data = np.zeros(shape).astype(np.float32)
    uncertainty = StdDevUncertainty(np.sqrt(np.abs(data)))
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.1, 0.1
    wcs.wcs.crpix = 0, 0
    wcs.wcs.crval = 1, 1

    if level in ["2", "3"] and code[0] == "P":
        wcs = add_stokes_axis_to_wcs(wcs, 2)
    wcs.array_shape = shape

    meta = NormalizedMetadata.load_template(code, level)
    meta['DATE-OBS'] = str(datetime(2024, 1, 1, 0, 0, 0))
    meta['DATE-BEG'] = str(datetime(2024, 1, 1, 0, 0, 0))
    meta['DATE-END'] = str(datetime(2024, 1, 1, 0, 0, 0))
    meta['DATE-AVG'] = str(datetime(2024, 1, 1, 0, 0, 0))
    meta['FILEVRSN'] = "1"
    meta['POLARREF'] = "Instrument"
    meta['POLAROFF'] = 0.0
    return NDCube(data=data, uncertainty=uncertainty, wcs=wcs, meta=meta)


def construct_all_product_headers(directory, level, outpath):
    date_obs = datetime.now()
    level_path = os.path.join(directory, f"Level{level}.yaml")
    level_spec = load_level_spec(level_path)
    product_keys = list(level_spec["Products"].keys())
    # crafts = load_spacecraft_def().keys()
    if level in ["0", "1", "2"]:
        crafts = {'1': '', '2': '', '3': '', '4': ''}.keys()
        shape = (2048,2048)
    if level in ["2", "3", "Q", "L"]:
        crafts = {'M': '', 'N': ''}.keys()
        shape = (4096,4096)
    if level in ["Q", "L"]:
        crafts = {'M': '', 'N': ''}.keys()
        shape = (1024,1024)
    product_keys = sorted(list(set([pc.replace("?", craft) for craft in crafts for pc in product_keys])))
    for pc in product_keys:
        try:
            meta = NormalizedMetadata.load_template(pc, level)
        except Exception as e:
            assert False, f"failed to create {pc} for level {level} because: {e}"
        meta['DATE-OBS'] = str(datetime.now())

        sample_data = sample_ndcube(shape=shape, code=pc, level=level)

        filename = outpath + get_base_file_name(sample_data) + '.fits'

        print('Finished writing ' + filename)

        write_ndcube_to_fits(sample_data, filename=filename, write_hash=False)


if __name__ == "__main__":

    path_yaml = '/Users/clowder/work/punch/punchbowl/punchbowl/data/data/'
    path_output = '/Users/clowder/data/punch/metadata/'

    for level in LEVELS:
        construct_all_product_headers(path_yaml, level, path_output)

    print("Job's finished.")
