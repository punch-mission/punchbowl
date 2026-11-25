import numpy as np

from punchbowl.data.tests.test_punch_io import sample_ndcube
from punchbowl.level1.alignment import (
    align_task,
    filter_for_visible_stars,
    find_catalog_in_image,
    load_gaia_catalog,
    prep_star_coords,
)


def test_late_core_flow_runs_with_filenames(sample_ndcube, tmpdir):
    sample_data = sample_ndcube(shape=(1024, 1024), code="XM1", level="1")
    for wcs in (sample_data.wcs, sample_data.celestial_wcs):
        # Make the WCS more realistic
        wcs.wcs.cdelt = -0.0225, 0.0225
        wcs.wcs.crpix = sample_data.shape[0] / 2, sample_data.shape[1] / 2
        wcs.wcs.ctype = ['RA---AZP', 'DEC--AZP']

    # Fill the sample cube with stars
    catalog_stars = filter_for_visible_stars(load_gaia_catalog(), dimmest_magnitude=7)
    catalog_stars = prep_star_coords(catalog_stars, sample_data.celestial_wcs)
    catalog_stars, pix_coords = find_catalog_in_image(catalog_stars, sample_data.celestial_wcs, sample_data.shape)
    yy, xx = np.indices(sample_data.shape)
    for x, y in zip(*pix_coords.T):
        gaussian = 100 * np.exp(-((xx-x)**2 + (yy-y)**2) / 2)
        sample_data.data += gaussian

    aligned_cube = align_task.fn(sample_data, distortion_path=None, max_workers=1, n_rounds=10)
    # Ensure alignment actually edited the WCS
    assert (sample_data.wcs.wcs.crval != aligned_cube.wcs.wcs.crval).all()
    assert (sample_data.celestial_wcs.wcs.crval != aligned_cube.celestial_wcs.wcs.crval).all()
