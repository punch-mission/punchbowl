import os
import json
from datetime import UTC, datetime, timedelta

import pytest
from prefect.testing.utilities import prefect_test_harness
from pytest_mock_resources import create_mysql_fixture

from punchbowl import __version__
from punchbowl.pipe.control.db import Base, File, Flow
from punchbowl.pipe.control.util import load_pipeline_configuration
from punchbowl.pipe.flows.level1 import (
    level1_early_construct_file_info,
    level1_early_construct_flow_info,
    level1_early_query_ready_files,
    level1_early_scheduler_flow,
    level1_late_construct_file_info,
    level1_late_construct_flow_info,
    level1_late_query_ready_files,
    level1_late_scheduler_flow,
    level1_middle_construct_file_info,
    level1_middle_construct_flow_info,
    level1_middle_query_ready_files,
    level1_middle_scheduler_flow,
)

TEST_DIR = os.path.dirname(__file__)

@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    with prefect_test_harness():
        yield

def session_fn(session):
    level0_file = File(level="0",
                       file_type='PM',
                       observatory='2',
                       state='created',
                       polarization='M',
                       file_version='none',
                       software_version='none',
                       crota=90,
                       date_obs=datetime.now(UTC))

    level1_file = File(level="1",
                       file_type="PM",
                       observatory='2',
                       state='created',
                       polarization='M',
                       file_version='none',
                       software_version='none',
                       date_obs=datetime.now(UTC))

    level1_x_file = File(level="1",
                       file_type="XM",
                       observatory='2',
                       state='created',
                       polarization='M',
                       file_version='none',
                       software_version='none',
                       date_obs=datetime.now(UTC))

    level1_y_file = File(level="1",
                       file_type="YM",
                       observatory='2',
                       state='created',
                       polarization='M',
                       file_version='none',
                       software_version='none',
                       date_obs=datetime.now(UTC))

    psf_model2 = File(level="1",
                       file_type="RM",
                       observatory='2',
                       state='created',
                       polarization='M',
                       file_version='0b',
                       software_version='none',
                       date_obs=datetime.now(UTC)-timedelta(hours=3))

    psf_model1 = File(level="1",
                       file_type="RM",
                       observatory='2',
                       state='created',
                       polarization='M',
                       file_version='1',
                       software_version='none',
                       date_obs=datetime.now(UTC)-timedelta(days=1))

    psf_model0 = File(level="1",
                       file_type="RM",
                       observatory='2',
                       state='created',
                       polarization='M',
                       file_version='0b',
                       software_version='none',
                       date_obs=datetime.now(UTC)-timedelta(days=1))

    quartic_fit_coeffs2 = File(level="1",
                       file_type="FQ",
                       observatory='2',
                       state='created',
                       file_version='0b',
                       software_version='none',
                       date_obs=datetime.now(UTC)-timedelta(hours=3))

    quartic_fit_coeffs1 = File(level="1",
                       file_type="FQ",
                       observatory='2',
                       state='created',
                       file_version='1',
                       software_version='none',
                       date_obs=datetime.now(UTC)-timedelta(days=1))

    quartic_fit_coeffs0 = File(level="1",
                       file_type="FQ",
                       observatory='2',
                       state='created',
                       file_version='0b',
                       software_version='none',
                       date_obs=datetime.now(UTC)-timedelta(days=1))

    vignetting_function2 = File(level="1",
                       file_type="GM",
                       observatory='2',
                       state='created',
                       file_version='0b',
                       software_version='none',
                       date_obs=datetime.now(UTC)-timedelta(hours=3))

    vignetting_function1 = File(level="1",
                       file_type="GM",
                       observatory='2',
                       state='created',
                       file_version='1',
                       software_version='none',
                       date_obs=datetime.now(UTC)-timedelta(days=1))

    vignetting_function0 = File(level="1",
                       file_type="GM",
                       observatory='2',
                       state='created',
                       file_version='0b',
                       software_version='none',
                       date_obs=datetime.now(UTC)-timedelta(days=1))

    stray_light_before2 = File(level="1",
                       file_type="SM",
                       observatory='2',
                       state='created',
                       polarization='M',
                       file_version='1',
                       software_version='none',
                       date_obs=datetime.now(UTC)-timedelta(hours=12))

    stray_light_before1 = File(level="1",
                       file_type="SM",
                       observatory='2',
                       state='created',
                       polarization='M',
                       file_version='2',
                       software_version='none',
                       date_obs=datetime.now(UTC)-timedelta(hours=16))

    stray_light_after1 = File(level="1",
                       file_type="SM",
                       observatory='2',
                       state='created',
                       polarization='M',
                       file_version='2',
                       software_version='none',
                       date_obs=datetime.now(UTC)+timedelta(hours=16))

    stray_light_after0 = File(level="1",
                       file_type="SM",
                       observatory='2',
                       state='created',
                       polarization='M',
                       file_version='1',
                       software_version='none',
                       date_obs=datetime.now(UTC)+timedelta(hours=12))

    dyn_stray_light_before2 = File(level="1",
                       file_type="TM",
                       observatory='2',
                       state='created',
                       polarization='M',
                       file_version='1',
                       software_version='none',
                       date_obs=datetime.now(UTC)-timedelta(hours=12))

    dyn_stray_light_before1 = File(level="1",
                       file_type="TM",
                       observatory='2',
                       state='created',
                       polarization='M',
                       file_version='2',
                       software_version='none',
                       date_obs=datetime.now(UTC)-timedelta(hours=16))

    dyn_stray_light_after1 = File(level="1",
                       file_type="TM",
                       observatory='2',
                       state='created',
                       polarization='M',
                       file_version='2',
                       software_version='none',
                       date_obs=datetime.now(UTC)+timedelta(hours=16))

    dyn_stray_light_after0 = File(level="1",
                       file_type="TM",
                       observatory='2',
                       state='created',
                       polarization='M',
                       file_version='1',
                       software_version='none',
                       date_obs=datetime.now(UTC)+timedelta(hours=12))

    distortion2 = File(level="1",
                       file_type="DS",
                       observatory='2',
                       state='created',
                       file_version='0b',
                       software_version='none',
                       date_obs=datetime.now(UTC)-timedelta(hours=3))

    distortion1 = File(level="1",
                       file_type="DS",
                       observatory='2',
                       state='created',
                       file_version='1',
                       software_version='none',
                       date_obs=datetime.now(UTC)-timedelta(days=1))

    distortion0 = File(level="1",
                       file_type="DS",
                       observatory='2',
                       state='created',
                       file_version='0b',
                       software_version='none',
                       date_obs=datetime.now(UTC)-timedelta(days=1))

    mask_file2 = File(level="1",
                       file_type="MS",
                       observatory='2',
                       state='created',
                       file_version='0b',
                       software_version='none',
                       date_obs=datetime.now(UTC)-timedelta(hours=3))

    mask_file1 = File(level="1",
                       file_type="MS",
                       observatory='2',
                       state='created',
                       file_version='1',
                       software_version='none',
                       date_obs=datetime.now(UTC)-timedelta(days=1))

    mask_file0 = File(level="1",
                       file_type="MS",
                       observatory='2',
                       state='created',
                       file_version='0b',
                       software_version='none',
                       date_obs=datetime.now(UTC)-timedelta(days=1))

    session.add(level0_file)
    session.add(level1_file)
    session.add(level1_x_file)
    session.add(level1_y_file)

    session.add(psf_model0)
    session.add(psf_model1)
    session.add(psf_model2)
    session.add(quartic_fit_coeffs0)
    session.add(quartic_fit_coeffs1)
    session.add(quartic_fit_coeffs2)
    session.add(vignetting_function0)
    session.add(vignetting_function1)
    session.add(vignetting_function2)
    session.add(distortion0)
    session.add(distortion1)
    session.add(distortion2)
    session.add(stray_light_before1)
    session.add(stray_light_before2)
    session.add(stray_light_after0)
    session.add(stray_light_after1)
    session.add(dyn_stray_light_before1)
    session.add(dyn_stray_light_before2)
    session.add(dyn_stray_light_after0)
    session.add(dyn_stray_light_after1)
    session.add(mask_file0)
    session.add(mask_file1)
    session.add(mask_file2)

db = create_mysql_fixture(Base, session_fn, session=True)


def test_level1_early_query_ready_files(db, prefect_test_fixture):
    pipeline_config_path = os.path.join(TEST_DIR, "punchpipe_config.yaml")
    pipeline_config = load_pipeline_configuration(pipeline_config_path)
    ready_file_ids = level1_early_query_ready_files(db, pipeline_config)
    assert len(ready_file_ids) == 1


def test_level1_middle_query_ready_files(db, prefect_test_fixture):
    pipeline_config_path = os.path.join(TEST_DIR, "punchpipe_config.yaml")
    pipeline_config = load_pipeline_configuration(pipeline_config_path)
    ready_file_ids = level1_middle_query_ready_files(db, pipeline_config)
    assert len(ready_file_ids) == 1


def test_level1_late_query_ready_files(db, prefect_test_fixture):
    pipeline_config_path = os.path.join(TEST_DIR, "punchpipe_config.yaml")
    pipeline_config = load_pipeline_configuration(pipeline_config_path)
    ready_file_ids = level1_late_query_ready_files(db, pipeline_config)
    assert len(ready_file_ids) == 1


def test_level1_early_construct_file_info():
    pipeline_config_path = os.path.join(TEST_DIR, "punchpipe_config.yaml")
    pipeline_config = load_pipeline_configuration(pipeline_config_path)
    level0_file = [File(level="0",
                       file_type='PM',
                       observatory='2',
                       state='created',
                       file_version='none',
                       software_version='none',
                       crota=90,
                       date_obs=datetime.now(UTC))]
    constructed_file_info = level1_early_construct_file_info(level0_file, pipeline_config)[0]
    assert constructed_file_info.level == "1"
    assert constructed_file_info.file_type == 'X' + level0_file[0].file_type[1:]
    assert constructed_file_info.observatory == level0_file[0].observatory
    assert constructed_file_info.file_version == "0.0.1"
    assert constructed_file_info.software_version == __version__
    assert constructed_file_info.date_obs == level0_file[0].date_obs
    assert constructed_file_info.polarization == level0_file[0].polarization
    assert constructed_file_info.state == "planned"


def test_level1_early_construct_file_info_clear():
    pipeline_config_path = os.path.join(TEST_DIR, "punchpipe_config.yaml")
    pipeline_config = load_pipeline_configuration(pipeline_config_path)
    level0_file = [File(level="0",
                       file_type='CR',
                       observatory='2',
                       state='created',
                       file_version='none',
                       software_version='none',
                       polarization='C',
                       crota=90,
                       date_obs=datetime.now(UTC))]
    x_file = level1_early_construct_file_info(level0_file, pipeline_config)[0]
    assert x_file.file_type == 'X' + level0_file[0].file_type[1:]
    assert x_file.level == "1"
    assert x_file.observatory == level0_file[0].observatory
    assert x_file.file_version == "0.0.1"
    assert x_file.software_version == __version__
    assert x_file.date_obs == level0_file[0].date_obs
    assert x_file.polarization == level0_file[0].polarization
    assert x_file.state == "planned"


def test_level1_middle_construct_file_info():
    pipeline_config_path = os.path.join(TEST_DIR, "punchpipe_config.yaml")
    pipeline_config = load_pipeline_configuration(pipeline_config_path)
    input_file = [File(level="0",
                       file_type='XM',
                       observatory='2',
                       state='created',
                       file_version='none',
                       software_version='none',
                       crota=90,
                       date_obs=datetime.now(UTC))]
    constructed_file_info = level1_middle_construct_file_info(input_file, pipeline_config)[0]
    assert constructed_file_info.level == "1"
    assert constructed_file_info.file_type == 'Y' + input_file[0].file_type[1:]
    assert constructed_file_info.observatory == input_file[0].observatory
    assert constructed_file_info.file_version == "0.0.1"
    assert constructed_file_info.software_version == __version__
    assert constructed_file_info.date_obs == input_file[0].date_obs
    assert constructed_file_info.polarization == input_file[0].polarization
    assert constructed_file_info.state == "planned"


def test_level1_late_construct_file_info():
    pipeline_config_path = os.path.join(TEST_DIR, "punchpipe_config.yaml")
    pipeline_config = load_pipeline_configuration(pipeline_config_path)
    input_file = [File(level="0",
                       file_type='YM',
                       observatory='2',
                       state='created',
                       file_version='none',
                       software_version='none',
                       crota=90,
                       date_obs=datetime.now(UTC))]
    constructed_file_info = level1_late_construct_file_info(input_file, pipeline_config)[0]
    assert constructed_file_info.level == "1"
    assert constructed_file_info.file_type == 'P' + input_file[0].file_type[1:]
    assert constructed_file_info.observatory == input_file[0].observatory
    assert constructed_file_info.file_version == "0.0.1"
    assert constructed_file_info.software_version == __version__
    assert constructed_file_info.date_obs == input_file[0].date_obs
    assert constructed_file_info.polarization == input_file[0].polarization
    assert constructed_file_info.state == "planned"


def test_level1_early_construct_flow_info(db, prefect_test_fixture):
    pipeline_config_path = os.path.join(TEST_DIR, "punchpipe_config.yaml")
    pipeline_config = load_pipeline_configuration(pipeline_config_path)
    level0_file = [File(level="0",
                       file_type='PM',
                       observatory='2',
                       state='created',
                       file_version='none',
                       software_version='none',
                       crota=90,
                       date_obs=datetime.now(UTC))]
    level0_file[0].quartic_model = level0_file[0]
    level0_file[0].vignetting_functions = level0_file[0], None
    level0_file[0].mask_file = level0_file[0]

    despike_neighbors = [File(level="0",
                       file_type=file_type,
                       observatory='2',
                       state='created',
                       file_version='none',
                       software_version='none',
                       crota=90,
                       date_obs=datetime.now(UTC)) for file_type in ["PM", "PZ", "PZ", "PP", "PP", "CR"]]

    level0_file[0].despike_neighbors = despike_neighbors

    level1_file = level1_early_construct_file_info(level0_file, pipeline_config)
    flow_info = level1_early_construct_flow_info(level0_file, level1_file, pipeline_config, session=db)

    assert flow_info.flow_type == 'level1_early'
    assert flow_info.state == "planned"
    assert flow_info.flow_level == "1"
    assert flow_info.priority == 6


def test_level1_middle_construct_flow_info(db, prefect_test_fixture):
    pipeline_config_path = os.path.join(TEST_DIR, "punchpipe_config.yaml")
    print(pipeline_config_path)
    pipeline_config = load_pipeline_configuration(pipeline_config_path)
    level0_file = [File(level="0",
                       file_type='XM',
                       observatory='2',
                       state='created',
                       file_version='none',
                       software_version='none',
                       crota=90,
                       date_obs=datetime.now(UTC))]

    level0_file[0].dynamic_stray_light = level0_file[0], level0_file[0]

    level1_file = level1_middle_construct_file_info(level0_file, pipeline_config)
    flow_info = level1_middle_construct_flow_info(level0_file, level1_file, pipeline_config, session=db)

    assert flow_info.flow_type == 'level1_middle'
    assert flow_info.state == "planned"
    assert flow_info.flow_level == "1"
    assert flow_info.priority == 6


def test_level1_late_construct_flow_info(db, prefect_test_fixture):
    pipeline_config_path = os.path.join(TEST_DIR, "punchpipe_config.yaml")
    print(pipeline_config_path)
    pipeline_config = load_pipeline_configuration(pipeline_config_path)
    level0_file = [File(level="0",
                       file_type='YM',
                       observatory='2',
                       state='created',
                       file_version='none',
                       software_version='none',
                       crota=90,
                       date_obs=datetime.now(UTC))]

    level0_file[0].psf_path = ""
    level0_file[0].distortion_path = level0_file[0]
    level0_file[0].stray_light = level0_file[0], level0_file[0]
    level0_file[0].mask_path = level0_file[0]

    level1_file = level1_late_construct_file_info(level0_file, pipeline_config)
    flow_info = level1_late_construct_flow_info(level0_file, level1_file, pipeline_config, session=db)

    assert flow_info.flow_type == 'level1_late'
    assert flow_info.state == "planned"
    assert flow_info.flow_level == "1"
    assert flow_info.priority == 6



def test_level1_early_scheduler_flow(db):
    pipeline_config_path = os.path.join(TEST_DIR, "punchpipe_config.yaml")
    with prefect_test_harness():
        level1_early_scheduler_flow(pipeline_config_path, db)
    results = db.query(Flow).where(Flow.state == 'planned').all()
    assert len(results) == 1

    call_data = json.loads(results[0].call_data)
    assert call_data["quartic_coefficient_path"][-7:] == 'v1.fits'
    assert call_data["vignetting_function_path"][-7:] == 'v1.fits'
    assert call_data["mask_path"][-6:] == 'v1.bin'



def test_level1_middle_scheduler_flow(db):
    pipeline_config_path = os.path.join(TEST_DIR, "punchpipe_config.yaml")
    with prefect_test_harness():
        level1_middle_scheduler_flow(pipeline_config_path, db)
    results = db.query(Flow).where(Flow.state == 'planned').all()
    assert len(results) == 1

    call_data = json.loads(results[0].call_data)
    assert call_data["dynamic_stray_light_before_path"][-7:] == 'v1.fits'
    assert call_data["dynamic_stray_light_after_path"][-7:] == 'v1.fits'


def test_level1_late_scheduler_flow(db):
    pipeline_config_path = os.path.join(TEST_DIR, "punchpipe_config.yaml")
    with prefect_test_harness():
        level1_late_scheduler_flow(pipeline_config_path, db)
    results = db.query(Flow).where(Flow.state == 'planned').all()
    assert len(results) == 1

    call_data = json.loads(results[0].call_data)
    assert call_data["psf_model_path"][-7:] == 'v1.fits'
    assert call_data["stray_light_before_path"][-7:] == 'v1.fits'
    assert call_data["stray_light_after_path"][-7:] == 'v1.fits'
    assert call_data["distortion_path"][-7:] == 'v1.fits'
    assert call_data["mask_path"][-6:] == 'v1.bin'


def test_level1_process_flow(db):
    pass
