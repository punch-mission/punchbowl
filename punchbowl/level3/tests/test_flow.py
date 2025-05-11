import os
import pathlib
from datetime import UTC, datetime, timedelta

import pytest
from ndcube import NDCube

from punchbowl.data.punch_io import write_ndcube_to_fits
from punchbowl.data.tests.test_punch_io import sample_ndcube
from punchbowl.level3.flow import level3_core_flow

THIS_DIRECTORY = pathlib.Path(__file__).parent.resolve()

# We run these tests first because when they were at the end they timed out on Prefect's fixture.
# Putting them first fixes that. Not sure why.

@pytest.mark.order(1)
def test_core_flow_runs_with_filenames(sample_ndcube, tmpdir):
    data_list = [sample_ndcube(shape=(3, 10, 10), code="PTM", level="2") for _ in range(12)]
    before_f_corona_model_path = os.path.join(tmpdir, "before_f_corona.fits")
    before_f_corona_model = sample_ndcube(shape=(3, 10, 10), code="PFM", level="3")
    before_f_corona_model.meta['DATE-OBS'] = str(datetime(2024, 2, 22, 16, 0, 1) - timedelta(hours=5))
    write_ndcube_to_fits(before_f_corona_model, before_f_corona_model_path)

    after_f_corona_model_path = os.path.join(tmpdir, "after_f_corona.fits")
    after_f_corona_model = sample_ndcube(shape=(3, 10, 10), code="PFM", level="3")
    after_f_corona_model.meta['DATE-OBS'] = str(datetime(2024, 2, 22, 16, 0, 1) + timedelta(hours=5))
    write_ndcube_to_fits(after_f_corona_model, after_f_corona_model_path)

    paths = []
    for i, cube in enumerate(data_list):
        path = os.path.join(tmpdir, f"test_input_{i}.fits")
        write_ndcube_to_fits(cube, path)
        paths.append(path)

    output = level3_core_flow(data_list,
                              before_f_corona_model_path, after_f_corona_model_path,
                              None)
    assert isinstance(output[0], NDCube)

@pytest.mark.order(1)
def test_core_flow_clear_runs_with_filenames(sample_ndcube, tmpdir):
    data_list = [sample_ndcube(shape=(10, 10), code="CTM", level="2") for _ in range(12)]
    before_f_corona_model_path = os.path.join(tmpdir, "before_f_corona.fits")
    before_f_corona_model = sample_ndcube(shape=(10, 10), code="CFM", level="3")
    before_f_corona_model.meta['DATE-OBS'] = str(datetime(2024, 2, 22, 16, 0, 1) - timedelta(hours=5))
    write_ndcube_to_fits(before_f_corona_model, before_f_corona_model_path)

    after_f_corona_model_path = os.path.join(tmpdir, "after_f_corona.fits")
    after_f_corona_model = sample_ndcube(shape=(10, 10), code="CFM", level="3")
    after_f_corona_model.meta['DATE-OBS'] = str(datetime(2024, 2, 22, 16, 0, 1) + timedelta(hours=5))
    write_ndcube_to_fits(after_f_corona_model, after_f_corona_model_path)

    paths = []
    for i, cube in enumerate(data_list):
        path = os.path.join(tmpdir, f"test_input_{i}.fits")
        write_ndcube_to_fits(cube, path)
        paths.append(path)

    output = level3_core_flow(data_list,
                              before_f_corona_model_path, after_f_corona_model_path,
                              None)
    assert isinstance(output[0], NDCube)

@pytest.mark.order(1)
@pytest.mark.parametrize("num_files", [1, 3])
def test_core_flow_runs_with_objects_and_calibration_files(tmpdir, sample_ndcube, num_files):
    data_list = [sample_ndcube(shape=(3, 10, 10), code="PTM", level="2") for _ in range(num_files)]

    before_f_corona_model_path = os.path.join(tmpdir, "before_f_corona.fits")
    before_f_corona_model = sample_ndcube(shape=(3, 10, 10), code="PFM", level="3")
    before_f_corona_model.meta['DATE-OBS'] =  str(datetime(2024, 2, 22, 16, 0, 1) - timedelta(hours=5))
    write_ndcube_to_fits(before_f_corona_model, before_f_corona_model_path)

    after_f_corona_model_path = os.path.join(tmpdir, "after_f_corona.fits")
    after_f_corona_model = sample_ndcube(shape=(3, 10, 10), code="PFM", level="3")
    after_f_corona_model.meta['DATE-OBS'] = str(datetime(2024, 2, 22, 16, 0, 1) + timedelta(hours=5))
    write_ndcube_to_fits(after_f_corona_model, after_f_corona_model_path)

    output = level3_core_flow(data_list,
                              before_f_corona_model_path, after_f_corona_model_path,
                              None)
    assert isinstance(output[0], NDCube)
