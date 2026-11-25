import os
import sys

import prefect.logging
import pytest

from punchbowl.data import get_base_file_name, write_ndcube_to_fits
from punchbowl.level2.nfi_pca import pca_filter
from punchbowl.levelq.tests.test_quickpunch_pca import generate_components, generate_images, generate_wcs


@pytest.mark.skipif(sys.version_info < (3, 13), reason="ShmPickleableArray requires Python 3.13 or higher")
def test_that_pca_filter_runs(tmpdir):
    components = generate_components()
    n_images = 30
    images = generate_images(components, n_images=n_images)

    image_paths = []
    for image in images:
        image_path = os.path.join(tmpdir, get_base_file_name(image) + '.fits')
        write_ndcube_to_fits(image, image_path, skip_stats=True, write_hash=False)
        image_paths.append(image_path)

    n_strides = 2
    n_components = 10
    with prefect.logging.disable_run_logger():
       output_cubes = pca_filter.fn(
            input_files=image_paths,
            context_files=[],
            nfi_mask=None,
            ref_date='2026-01-01T00:00:00',
            n_components=n_components,
            n_strides=n_strides,
            downsample_factor=2,
            n_workers=1,
            n_loaders=1,
        )
    assert len(output_cubes) == n_images + 2
    sr_cubes = [cube for cube in output_cubes if cube.meta['TYPECODE'].value == 'SR']
    ar_cubes = [cube for cube in output_cubes if cube.meta['TYPECODE'].value == 'AR']
    cnn_cubes = [cube for cube in output_cubes if cube.meta['TYPECODE'].value == 'CN']

    assert len(sr_cubes) == 1
    assert sr_cubes[0].meta['LEVEL'].value == '1'

    assert len(ar_cubes) == 1
    assert ar_cubes[0].meta['LEVEL'].value == '1'
    assert ar_cubes[0].shape[:2] == (n_strides, n_components + 1)

    assert len(cnn_cubes) == n_images
    assert all(cube.meta['OBSCODE'].value == 'N' for cube in cnn_cubes)
    assert all(cube.meta['LEVEL'].value == '2' for cube in cnn_cubes)
    assert cnn_cubes[0].shape == images[0].shape
