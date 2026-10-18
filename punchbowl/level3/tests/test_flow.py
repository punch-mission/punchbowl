import os
import pathlib
from datetime import UTC, datetime, timedelta

import pytest
from ndcube import NDCube
from prefect.testing.utilities import prefect_test_harness

from punchbowl.conftest import prefect_test_fixture
from punchbowl.data.punch_io import write_ndcube_to_fits
from punchbowl.data.tests.test_punch_io import sample_ndcube
from punchbowl.level3.flow import level3_PIM_CIM_flow

THIS_DIRECTORY = pathlib.Path(__file__).parent.resolve()


@pytest.mark.parametrize('as_cubes', [True, False])
def test_PIM_flow_runs_with_filenames(sample_ndcube, tmpdir, prefect_test_fixture, as_cubes):
    data_list = [sample_ndcube(shape=(3, 10, 10), code=f"XP{obs + 1}", level="2") for obs in range(4)]
    for c in data_list:
        c.meta['CROPX2'] = 10
        c.meta['CROPY2'] = 10
        c.meta['FULXSIZE'] = 10
        c.meta['FULYSIZE'] = 10

    before_models = []
    for i, f in enumerate(data_list):
        before_f_corona_model = sample_ndcube(shape=(3, 10, 10), code="PF" + f.meta['OBSCODE'].value, level="3")
        before_f_corona_model_path = os.path.join(tmpdir, f"before_f_corona_{i}.fits")
        before_f_corona_model.meta['DATE-OBS'] = str(datetime(2024, 2, 22, 16, 0, 1) - timedelta(hours=5))
        before_models.append(before_f_corona_model_path)
        write_ndcube_to_fits(before_f_corona_model, before_f_corona_model_path, write_hash=False, skip_stats=True)

    after_models = []
    for i, f in enumerate(data_list):
        after_f_corona_model = sample_ndcube(shape=(3, 10, 10), code="PF" + f.meta['OBSCODE'].value, level="3")
        after_f_corona_model_path = os.path.join(tmpdir, f"after_f_corona_{i}.fits")
        after_f_corona_model.meta['DATE-OBS'] = str(datetime(2024, 2, 22, 16, 0, 1) + timedelta(hours=5))
        after_models.append(after_f_corona_model_path)
        write_ndcube_to_fits(after_f_corona_model, after_f_corona_model_path, write_hash=False, skip_stats=True)

    if not as_cubes:
        paths = []
        for i, cube in enumerate(data_list):
            path = os.path.join(tmpdir, f"test_input_{i}.fits")
            write_ndcube_to_fits(cube, path)
            paths.append(path)
        data_list = paths

    output = level3_PIM_CIM_flow(data_list,
                                 before_models,
                                 after_models,
                                 None)
    assert isinstance(output[0], NDCube)
    assert output[0].meta['TYPECODE'].value == 'PI'
    assert output[0].meta['OBSCODE'].value == 'M'


def test_PIM_flow_clear_runs_with_filenames(sample_ndcube, tmpdir, prefect_test_fixture):
    data_list = [sample_ndcube(shape=(10, 10), code=f"XR{obs + 1}", level="2") for obs in range(4)]
    for c in data_list:
        c.meta['CROPX2'] = 10
        c.meta['CROPY2'] = 10
        c.meta['FULXSIZE'] = 10
        c.meta['FULYSIZE'] = 10

    before_models = []
    for i, f in enumerate(data_list):
        before_f_corona_model = sample_ndcube(shape=(10, 10), code="CF" + f.meta['OBSCODE'].value, level="3")
        before_f_corona_model_path = os.path.join(tmpdir, f"before_f_corona_{i}.fits")
        before_f_corona_model.meta['DATE-OBS'] = str(datetime(2024, 2, 22, 16, 0, 1) - timedelta(hours=5))
        before_models.append(before_f_corona_model_path)
        write_ndcube_to_fits(before_f_corona_model, before_f_corona_model_path, write_hash=False, skip_stats=True)

    after_models = []
    for i, f in enumerate(data_list):
        after_f_corona_model = sample_ndcube(shape=(10, 10), code="CF" + f.meta['OBSCODE'].value, level="3")
        after_f_corona_model_path = os.path.join(tmpdir, f"after_f_corona_{i}.fits")
        after_f_corona_model.meta['DATE-OBS'] = str(datetime(2024, 2, 22, 16, 0, 1) + timedelta(hours=5))
        after_models.append(after_f_corona_model_path)
        write_ndcube_to_fits(after_f_corona_model, after_f_corona_model_path, write_hash=False, skip_stats=True)

    paths = []
    for i, cube in enumerate(data_list):
        path = os.path.join(tmpdir, f"test_input_{i}.fits")
        write_ndcube_to_fits(cube, path)
        paths.append(path)

    output = level3_PIM_CIM_flow(data_list,
                                 before_models,
                                 after_models,
                                 None)
    assert isinstance(output[0], NDCube)
    assert output[0].meta['TYPECODE'].value == 'CI'
    assert output[0].meta['OBSCODE'].value == 'M'
