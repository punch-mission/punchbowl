import numpy as np
import prefect.logging
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube

from punchbowl.data import NormalizedMetadata
from punchbowl.levelq.pca import find_bodies_in_image_quarters, pca_filter


def test_find_bodies_in_image_quarters():
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = 'HPLN-ARC', 'HPLT-ARC'
    # These are values copied from a real image
    wcs.wcs.crval = 0.11492928582807, 0.088699676942111
    wcs.wcs.crpix = 1024.5, 1024.5
    wcs.wcs.cdelt = 0.008333333333, 0.008333333333
    wcs.wcs.dateobs = '2025-07-05T12:56:21.706'

    meta = NormalizedMetadata.load_template('CR4', '1')
    meta['GEOD_LON'] = 75.80763246719741
    meta['GEOD_LAT'] = 2.6530022837437506
    meta['GEOD_ALT'] = 642990.3420114828

    cube = NDCube(data=np.empty((2048, 2048)), meta=meta, wcs=wcs)

    result = find_bodies_in_image_quarters(cube)

    # This should come out as all False except for one spot
    assert result[4][3]
    result[4][3] = False
    assert np.all(np.array(result) == False)


def test_that_pca_filter_runs():
    cubes = []

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = 'HPLN-ARC', 'HPLT-ARC'
    # These are values copied from a real image
    wcs.wcs.crval = 0.11492928582807, 0.088699676942111
    wcs.wcs.crpix = 128.5, 128.5
    wcs.wcs.cdelt = 0.0664, 0.0664
    wcs.wcs.dateobs = '2025-07-05T12:56:21.706'
    wcs.array_shape = (256, 256)

    meta = NormalizedMetadata.load_template('CR4', '1')
    meta['GEOD_LON'] = 75.80763246719741
    meta['GEOD_LAT'] = 2.6530022837437506
    meta['GEOD_ALT'] = 642990.3420114828

    uncertainty = np.ones((256, 256))
    # Generate input files that rotate so the same planet isn't in the same quarter for every image
    for i in np.linspace(0, 2*np.pi, 300):
        wcs.wcs.pc = np.array([[np.cos(i), -np.sin(i)], [np.sin(i), np.cos(i)]])
        cube = NDCube(data=np.ones((256, 256)), meta=meta, wcs=wcs.deepcopy(),
                      uncertainty=StdDevUncertainty(uncertainty))
        cubes.append(cube)

    with prefect.logging.disable_run_logger():
        pca_filter.fn(cubes, [], n_components=20, med_filt=3, n_strides=2, blend_size=5)
