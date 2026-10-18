import os
import sys
from datetime import datetime, timedelta

import numpy as np
import prefect.logging
import pytest
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube

from punchbowl.data import NormalizedMetadata, get_base_file_name, write_ndcube_to_fits
from punchbowl.levelq.quickpunch_pca import quickpunch_pca_filter

test_image_size = 128

def generate_wcs():
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = 'HPLN-ARC', 'HPLT-ARC'
    # These are values copied from a real image
    wcs.wcs.crval = 0.11492928582807, 0.088699676942111
    wcs.wcs.crpix = 64.5, 64.5
    wcs.wcs.cdelt = 0.1328, 0.1328
    wcs.wcs.dateobs = '2026-01-01T00:00:00'
    wcs.array_shape = (test_image_size, test_image_size)
    wcs.wcs.aux.hgln_obs = 0
    wcs.wcs.aux.hglt_obs = 7.19
    wcs.wcs.aux.dsun_obs = 150000000000
    return wcs

def generate_components():
    components = np.zeros((4, test_image_size, test_image_size), dtype=np.float32)
    components[1][30:40, 35:45] = 1
    components[2][60:80, 30:35] = 1
    components[3][50:60, 70:80] = 1

    components = components.reshape((1, 4, test_image_size * test_image_size))

    new_meta = NormalizedMetadata.load_template("AR4", "1")
    new_meta["DATE-OBS"] = '2026-01-01T00:00:00'
    new_meta["PCANCOMP"] = len(components) - 1
    new_meta["PCADWNSP"] = 1
    new_meta["FILEVRSN"] = 'test'
    pca_cube = NDCube(data=components, meta=new_meta, wcs=generate_wcs())
    return pca_cube

def generate_inst_frame_bg():
    bg = np.zeros((test_image_size, test_image_size))

    new_meta = NormalizedMetadata.load_template("SR4", "1")
    new_meta["DATE-OBS"] = '2026-01-01T00:00:00'
    new_meta["DATE-END"] = new_meta["DATE-OBS"].value
    new_meta["FILEVRSN"] = 'test'
    bg_cube = NDCube(data=bg, meta=new_meta, wcs=generate_wcs())
    return bg_cube

def generate_sky_frame_bg():
    bg = np.zeros((test_image_size, test_image_size))

    new_meta = NormalizedMetadata.load_template("CF4", "3")
    new_meta["DATE-OBS"] = '2026-01-01T00:00:00'
    new_meta["DATE-END"] = new_meta["DATE-OBS"].value
    new_meta["FILEVRSN"] = 'test'
    bg_cube = NDCube(data=bg, meta=new_meta, wcs=generate_wcs())
    return bg_cube

def generate_images(pca_components, n_images=10):
    components = pca_components.data.reshape((4, test_image_size, test_image_size))
    images = []
    for i, (a1, a2, a3) in enumerate(zip(range(-5, -5 + n_images), range(2, 2 + n_images), range(3 + n_images, 3, -1))):
        image = a1 * components[1] + a2 * components[2] + a3 * components[3] + 0.1
        uncertainty = StdDevUncertainty(np.full_like(image, .1))

        new_meta = NormalizedMetadata.load_template("XR4", "1")
        date_obs = datetime(2026, 1, 1) + timedelta(minutes=8) * i
        new_meta["DATE-OBS"] = date_obs.strftime("%Y-%m-%dT%H:%M:%S")
        new_meta["FILEVRSN"] = 'test'
        new_meta["BADPKTS"] = 0
        new_meta['DSATVAL'] = 999999999 # Don't let any pixels get flagged
        new_meta["DATAP25"] = 0 # This needs to be below a threshold to not be flagged as a bad image
        new_meta["DATAP98"] = a1 # These go into std-based outlier flagging, so they need some variation between images
        new_meta["DATAP90"] = a1
        new_meta["DATAP50"] = a1
        # These are values copied from a real image
        new_meta['GEOD_LON'] = -106.64419642829114
        new_meta['GEOD_LAT'] = -62.29165706219033
        new_meta['GEOD_ALT'] = 671813.7879664664
        new_meta['CRLT_OBS'] = -2.9996342326996883
        new_meta['CRLN_OBS'] = 299.10479748585095
        new_meta['HGLT_OBS'] = -0.00129722053856085
        new_meta['HGLN_OBS'] = 299.10479748585095
        new_meta['DSUN_OBS'] = 147103575649.20935
        images.append(NDCube(data=image, meta=new_meta, wcs=generate_wcs(), uncertainty=uncertainty))
    return images


@pytest.mark.skipif(sys.version_info < (3, 13), reason="ShmPickleableArray requires Python 3.13 or higher")
def test_that_pca_filter_runs(tmpdir):
    components = generate_components()
    images = generate_images(components)
    inst_frame_bg = generate_inst_frame_bg()
    first_sky_frame_bg = generate_sky_frame_bg()
    second_sky_frame_bg = generate_sky_frame_bg()
    first_sky_frame_bg.meta['DATE-OBS'] = '2025-12-31T00:00:00'
    second_sky_frame_bg.meta['DATE-OBS'] = '2026-01-02T00:00:00'

    inst_frame_bg_path = os.path.join(tmpdir, get_base_file_name(inst_frame_bg) + '.fits')
    write_ndcube_to_fits(
        inst_frame_bg, inst_frame_bg_path, skip_stats=True, write_hash=False)

    first_sky_bg_path = os.path.join(tmpdir, get_base_file_name(first_sky_frame_bg) + '.fits')
    write_ndcube_to_fits(
        first_sky_frame_bg, first_sky_bg_path, skip_stats=True, write_hash=False)

    second_sky_bg_path = os.path.join(tmpdir, get_base_file_name(second_sky_frame_bg) + '.fits')
    write_ndcube_to_fits(
        second_sky_frame_bg, second_sky_bg_path, skip_stats=True, write_hash=False)

    inst_frame_bg_path = os.path.join(tmpdir, get_base_file_name(inst_frame_bg) + '.fits')
    (write_ndcube_to_fits
     (inst_frame_bg, inst_frame_bg_path, skip_stats=True, write_hash=False))

    component_path = os.path.join(tmpdir, get_base_file_name(components) + '.fits')
    write_ndcube_to_fits(
        components, component_path, skip_stats=True, write_hash=False)

    image_paths = []
    for image in images:
        image_path = os.path.join(tmpdir, get_base_file_name(image) + '.fits')
        write_ndcube_to_fits(image, image_path, skip_stats=True, write_hash=False)
        image_paths.append(image_path)

    with prefect.logging.disable_run_logger():
       output_cubes = quickpunch_pca_filter.fn(
            input_files=image_paths,
            context_files=[],
            nfi_mask=None,
            pca_components_path=component_path,
            instrument_frame_background_path=inst_frame_bg_path,
            first_helio_frame_background_path=first_sky_bg_path,
            second_helio_frame_background_path=second_sky_bg_path,
            median_window=3,
            zfilter_margin=1,
            zfilter_index=0.5,
            n_workers=1,
            n_loaders=1,
        )
    assert len(output_cubes) == 7
    assert all(cube.meta['TYPECODE'].value == 'QN' for cube in output_cubes)
    assert all(cube.meta['OBSCODE'].value == 'N' for cube in output_cubes)
    assert all(cube.meta['LEVEL'].value == 'Q' for cube in output_cubes)
