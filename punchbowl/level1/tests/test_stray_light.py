import os
import pathlib
from datetime import datetime

from ndcube import NDCube
from prefect.logging import disable_run_logger

from punchbowl.data import write_ndcube_to_fits
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
