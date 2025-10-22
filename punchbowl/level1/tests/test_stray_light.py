import os
import pathlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from ndcube import NDCube
from prefect.logging import disable_run_logger

from punchbowl.data import NormalizedMetadata, punch_io, write_ndcube_to_fits
from punchbowl.data.tests.test_punch_io import sample_ndcube
from punchbowl.level1.stray_light import estimate_polarized_stray_light, estimate_stray_light, remove_stray_light_task

THIS_DIRECTORY = pathlib.Path(__file__).parent.resolve()


def test_no_straylight_file(sample_ndcube) -> None:
    """
    An invalid vignetting file should be provided. Check that an error is raised.
    """

    sample_data = sample_ndcube(shape=(10, 10), code="CR1", level="0")
    straylight_before_filename = None
    straylight_after_filename = None

    with disable_run_logger():
        corrected_punchdata = remove_stray_light_task.fn(sample_data, straylight_before_filename, straylight_after_filename)
        assert isinstance(corrected_punchdata, NDCube)
        assert corrected_punchdata.meta.history[0].comment == 'Stray light correction skipped'


def test_estimate_stray_light_runs(tmpdir, sample_ndcube):
    data_list = [sample_ndcube(shape=(10, 10), code='XR1', level="1") for i in range(10)]

    paths = []
    for i, cube in enumerate(data_list):
        path = os.path.join(tmpdir, f"test_input_{i}.fits")
        write_ndcube_to_fits(cube, path)
        paths.append(path)

    with disable_run_logger():
        cube = estimate_stray_light.fn(paths, 3)

    assert cube[0].meta['TYPECODE'].value == 'SR'
    assert cube[0].meta['OBSCODE'].value == '1'


@pytest.fixture
def dummy_fits_paths(tmp_path: Path):
    """Create dummy fits files for testing."""
    values = [10.0, 5.0, 25.0]

    out_lists = {"m": [], "z": [], "p": []}

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.cdelt = 0.01, 0.01
    wcs.wcs.crpix = 1024, 1024
    wcs.wcs.crval = 0, 0
    wcs.wcs.cname = "HPC lon", "HPC lat"
    wcs.array_shape = (2048, 2048)

    for i, val in enumerate(values):
        arr = np.full((3, 3), val, dtype=float)

        for prefix in ["m", "z", "p"]:
            path = tmp_path / f"{prefix}{i}.fits"

            meta = NormalizedMetadata.load_template(product_code=f"P{prefix.upper()}1", level="1")
            meta["DATE-OBS"] = f"2008-01-03 0{i}:57:00"
            cube = NDCube(data=arr, uncertainty=None, wcs=wcs, meta=meta)
            write_ndcube_to_fits(cube, str(path))

            out_lists[prefix].append(str(path))

    return out_lists['m'], out_lists['z'], out_lists['p']

def test_estimate_polarized_stray_light(dummy_fits_paths) -> None:
    mfiles, zfiles, pfiles = dummy_fits_paths
    with disable_run_logger():
        result = estimate_polarized_stray_light.fn(mfiles, zfiles, pfiles, do_uncertainty=False)
        for cube in result:
            assert isinstance(cube, NDCube)
            assert cube.data.shape == (3, 3)
            assert np.allclose(cube.data, 5.0)


def test_estimate_polarized_stray_light_runs(tmpdir, sample_ndcube):
    dates = [datetime(2025, 1, d, 1, 2, 3).strftime('%Y-%m-%dT%H:%M:%S') for d in range(1, 11)]
    m_cubes = [sample_ndcube(shape=(10, 10), code='XM1', level="1", date_obs=date) for date in dates]
    z_cubes = [sample_ndcube(shape=(10, 10), code='XZ1', level="1", date_obs=date) for date in dates]
    p_cubes = [sample_ndcube(shape=(10, 10), code='XP1', level="1", date_obs=date) for date in dates]

    mpaths, zpaths, ppaths = [], [], []
    for i, cube in enumerate(m_cubes):
        path = os.path.join(tmpdir, f"test_input_M_{i}.fits")
        write_ndcube_to_fits(cube, path)
        mpaths.append(path)
    for i, cube in enumerate(z_cubes):
        path = os.path.join(tmpdir, f"test_input_Z_{i}.fits")
        write_ndcube_to_fits(cube, path)
        zpaths.append(path)
    for i, cube in enumerate(p_cubes):
        path = os.path.join(tmpdir, f"test_input_P_{i}.fits")
        write_ndcube_to_fits(cube, path)
        ppaths.append(path)

    with disable_run_logger():
        cubes = estimate_polarized_stray_light.fn(mpaths, zpaths, ppaths, num_loaders=1)

    assert cubes[0].meta['TYPECODE'].value == 'SM'
    assert cubes[1].meta['TYPECODE'].value == 'SZ'
    assert cubes[2].meta['TYPECODE'].value == 'SP'
    for cube in cubes:
        assert cube.meta['OBSCODE'].value == '1'
