import os
from datetime import UTC, datetime, timedelta

from freezegun import freeze_time
from prefect.logging import disable_run_logger
from pytest_mock_resources import create_mysql_fixture

from punchbowl import __version__
from punchbowl.auto.control.db import Base, File, FileRelationship
from punchbowl.auto.control.util import load_pipeline_configuration
from punchbowl.auto.flows.levelq import (
    levelq_CQM_construct_file_info,
    levelq_CQM_construct_flow_info,
    levelq_CQM_query_ready_files,
    levelq_CTM_construct_file_info,
    levelq_CTM_construct_flow_info,
    levelq_CTM_query_ready_files,
    levelq_QNN_query_ready_files,
)

TEST_DIR = os.path.dirname(__file__)


def session_fn(session):
    level0_file = File(level="0",
                       file_type="CR",
                       observatory="1",
                       state="progressed",
                       file_version="none",
                       software_version="none",
                       date_obs=datetime(2023, 1, 1, 0, 0, 0))

    level1_file_not_ready = File(level="1",
                                 file_type="QR",
                                 observatory="1",
                                 state="planned",
                                 file_version="none",
                                 software_version="none",
                                 date_obs=datetime(2023, 1, 1, 0, 0, 0))

    level1_file = File(level="1",
                       file_type="QR",
                       observatory="1",
                       state="created",
                       file_version="none",
                       software_version="none",
                       date_obs=datetime(2023, 1, 1, 0, 0, 0))

    levelQ_file = File(level="Q",
                       file_type="CQ",
                       observatory="M",
                       state="created",
                       file_version="none",
                       software_version="none",
                       date_obs=datetime(2023, 1, 1, 0, 0, 0))

    f_corona_before_file = File(level="Q",
                       file_type="CF",
                       observatory="M",
                       state="created",
                       file_version="none",
                       software_version="none",
                       date_obs=datetime(2022, 12, 31, 0, 0, 0))

    f_corona_after_file = File(level="Q",
                       file_type="CF",
                       observatory="M",
                       state="created",
                       file_version="none",
                       software_version="none",
                       date_obs=datetime(2023, 1, 2, 0, 0, 0))

    nfi_f_corona_before_file = File(level="3",
                                    file_type="CF",
                                    observatory="N",
                                    state="created",
                                    file_version="none",
                                    software_version="none",
                                    date_obs=datetime(2023, 1, 2, 0, 0, 0))

    nfi_f_corona_after_file = File(level="3",
                                   file_type="CF",
                                   observatory="N",
                                   state="created",
                                   file_version="none",
                                   software_version="none",
                                   date_obs=datetime(2023, 1, 3, 0, 0, 0))

    nfi_sl_file = File(level="1",
                       file_type="SR",
                       observatory="4",
                       state="created",
                       file_version="none",
                       software_version="none",
                       date_obs=datetime(2023, 1, 3, 0, 0, 0))

    nfi_pca_comp_file = File(level="1",
                             file_type="AR",
                             observatory="4",
                             state="created",
                             file_version="none",
                             software_version="none",
                             date_obs=datetime(2023, 1, 3, 0, 0, 0))

    session.add(level0_file)
    session.add(level1_file_not_ready)
    session.add(level1_file)
    session.add(levelQ_file)
    session.add(f_corona_before_file)
    session.add(f_corona_after_file)
    session.add(nfi_f_corona_before_file)
    session.add(nfi_f_corona_after_file)
    session.add(nfi_sl_file)
    session.add(nfi_pca_comp_file)


db = create_mysql_fixture(Base, session_fn, session=True)


def test_levelq_CNN_query_ready_files(db):
    pipeline_config = {'flows': {'levelq_QNN': {"batch_size_cap": 80,
                                                "n_batches_to_schedule": 1,
                                                "max_gap_minutes": 17,
                                                "median_window": 5,
                                                "zfilter_margin": 4,
                                                "only_last_n_days": 10}}}
    t0 = datetime.now() - timedelta(days=5)
    files = []
    for i in range(200):
        t = t0 + timedelta(minutes=i * 8)
        file = File(level="1",
                    file_type="XR",
                    observatory="4",
                    state="created",
                    file_version="none",
                    software_version="none",
                    date_obs=t,
                    date_created=t - timedelta(days=1),
                    file_id=i + 1000)
        files.append(file)
        if i == 198:
            final_gap_file1 = file
        elif i == 197:
            final_gap_file2 = file
        elif i == 50:
            mid_gap_file1 = file
        elif i == 51:
            mid_gap_file1 = file
        elif i == 20:
            ok_gap_file = file
        elif i == 70:
            missing_neighbor_file = file
        elif i == 100:
            other_missing_neighbor_file = file
        else:
            if i in (71, 99):
                file.date_created = datetime.now()
            if i == 60:
                already_processed_file = file
                child_file = File(level="Q",
                    file_type="QN",
                    observatory="N",
                    state="created",
                    file_version="none",
                    software_version="none",
                    date_obs=t,
                    date_created=t - timedelta(days=1),
                    file_id=99999)
                db.add(child_file)
                rel = FileRelationship(child=child_file.file_id, parent=file.file_id)
                db.add(rel)
            db.add(file)
    db.commit()

    """
         20        50,51        70,71                     99,100                                              197,198
    |----------------|------------|--------------------------|----------------|-----------------------------------|---|
    |     ^          ^            ^                          ^                ^                                   ^
          |          |            |            100 missing, 99 new,     break b/c group too large                 |
    missing, no gap  |            |               group break                                                 2 missing
         2 missing, group break   |                                                          group break, no 199-200 grp
         70 missing, 71 new, group break
    """
    groups = levelq_QNN_query_ready_files(db, pipeline_config)
    assert len(groups) == 1
    assert groups[0][0].file_id == 109 + 1000
    assert groups[0][-1].file_id == 199 + 1000

    pipeline_config['flows']['levelq_QNN']['n_batches_to_schedule'] = 10
    groups = levelq_QNN_query_ready_files(db, pipeline_config)
    assert len(groups) == 5

    assert groups[4][0].file_id == 0 + 1000
    assert groups[4][-1].file_id == 53 + 1000

    # split, 2 missing files

    assert groups[3][0].file_id == 44 + 1000
    assert groups[3][-2].file_id == 72 + 1000
    already_processed = [f for f in groups[3] if f.file_id == already_processed_file.file_id]
    assert len(already_processed) == 1
    assert not already_processed[0]._to_filter

    # split, 1 missing file and a neighbor is new

    assert groups[2][0].file_id == 63 + 1000
    assert groups[2][-1].file_id == 102 + 1000

    # split, 1 missing file and a neighbor is new

    assert groups[1][0].file_id == 93 + 1000
    assert groups[1][-1].file_id == 120 + 1000

    # split b/c following group got too large

    assert groups[0][0].file_id == 109 + 1000
    assert groups[0][-1].file_id == 199 + 1000


def test_levelq_CQM_query_ready_files(db):
    with disable_run_logger():
        with freeze_time(datetime(2023, 1, 1, 0, 5, 0)) as frozen_datatime:  # noqa: F841
            pipeline_config = {'flows': {'levelq_CQM': {"production_mode_max_wait_hours": 24,
                                                        "production_mode_cutoff_date": "2023-01-01"}}}
            ready_file_ids = levelq_CQM_query_ready_files.fn(db, pipeline_config)
            # Only one of the expected input files is ready
            assert len(ready_file_ids) == 0


def test_levelq_CTM_query_ready_files(db):
    with disable_run_logger():
        with freeze_time(datetime(2023, 1, 1, 0, 5, 0)) as frozen_datatime:  # noqa: F841
            pipeline_config = {'flows': {'levelq_CTM': {}}}
            ready_file_ids = levelq_CTM_query_ready_files.fn(db, pipeline_config)
            assert len(ready_file_ids) == 1


def test_levelq_CQM_query_ready_files_unprocessed_L0(db):
    try:
        with disable_run_logger(), freeze_time(datetime(2023, 1, 3, 0, 0, 0)):  # noqa: F841
            pipeline_config = {'flows': {'levelq_CQM': {"production_mode_max_wait_hours": 24,
                                                        "production_mode_cutoff_date": "2023-01-02"}}}
            ready_file_ids = levelq_CQM_query_ready_files.fn(db, pipeline_config)
            # We're missing imagers, but we have one for every L0
            assert len(ready_file_ids) == 1

            level0_file = File(level="0",
                               file_type="CR",
                               observatory="2",
                               state="progressed",
                               file_version="none",
                               software_version="none",
                               date_obs=datetime(2023, 1, 1, 0, 0, 0))
            db.add(level0_file)

            ready_file_ids = levelq_CQM_query_ready_files.fn(db, pipeline_config)
            # Now we have unprocessed L0s, so we shouldn't make anything
            assert len(ready_file_ids) == 0
    finally:
        db.rollback()


def test_levelq_CQM_query_ready_files_ignore_missing(db):
    with disable_run_logger():
        with freeze_time(datetime(2023, 1, 2, 0, 0, 0, tzinfo=UTC)) as frozen_datatime:  # noqa: F841
            pipeline_config = {'flows': {'levelq_CQM': {"production_mode_max_wait_hours": 25,
                                                        "production_mode_cutoff_date": "2023-01-01"}}}
            ready_file_ids = levelq_CQM_query_ready_files.fn(db, pipeline_config)
            assert len(ready_file_ids) == 0
            pipeline_config['flows']['levelq_CQM']['production_mode_max_wait_hours'] = 23
            ready_file_ids = levelq_CQM_query_ready_files.fn(db, pipeline_config)
            assert len(ready_file_ids) == 1


def test_levelq_CQM_construct_file_info():
    pipeline_config_path = os.path.join(TEST_DIR, "punchpipe_config.yaml")
    pipeline_config = load_pipeline_configuration(pipeline_config_path)
    level1_file = [File(level='1',
                       file_type='CR',
                       observatory='1',
                       state='created',
                       file_version='none',
                       software_version='none',
                       date_obs=datetime.now(UTC))]
    constructed_file_info = levelq_CQM_construct_file_info(level1_file, pipeline_config)[0]
    assert constructed_file_info.level == "Q"
    assert constructed_file_info.file_type == "CQ"
    assert constructed_file_info.observatory == "M"
    assert constructed_file_info.file_version == "0.0.1"
    assert constructed_file_info.software_version == __version__
    assert constructed_file_info.date_obs == level1_file[0].date_obs
    assert constructed_file_info.polarization == 'C'
    assert constructed_file_info.state == "planned"


def test_levelq_CTM_construct_file_info():
    pipeline_config_path = os.path.join(TEST_DIR, "punchpipe_config.yaml")
    pipeline_config = load_pipeline_configuration(pipeline_config_path)
    levelQ_file = [File(level='Q',
                        file_type='CQ',
                        observatory='M',
                        state='created',
                        file_version='none',
                        software_version='none',
                        date_obs=datetime.now(UTC))]
    constructed_file_info = levelq_CTM_construct_file_info(levelQ_file, pipeline_config)[0]
    assert constructed_file_info.level == "Q"
    assert constructed_file_info.file_type == "CT"
    assert constructed_file_info.observatory == "M"
    assert constructed_file_info.file_version == "0.0.1"
    assert constructed_file_info.software_version == __version__
    assert constructed_file_info.date_obs == levelQ_file[0].date_obs
    assert constructed_file_info.polarization == 'C'
    assert constructed_file_info.state == "planned"


def test_levelq_CQM_construct_flow_info():
    pipeline_config_path = os.path.join(TEST_DIR, "punchpipe_config.yaml")
    pipeline_config = load_pipeline_configuration(pipeline_config_path)
    input_files = [File(level='1',
                       file_type='CT',
                       observatory='2',
                       state='created',
                       file_version='none',
                       software_version='none',
                       date_obs=datetime.now(UTC))]
    output_file = levelq_CQM_construct_file_info(input_files, pipeline_config)
    flow_info = levelq_CQM_construct_flow_info(input_files, output_file[0], pipeline_config)

    assert flow_info.flow_type == 'levelq_CQM'
    assert flow_info.state == "planned"
    assert flow_info.flow_level == "Q"
    assert flow_info.priority == 1000


def test_levelq_CTM_construct_flow_info():
    pipeline_config_path = os.path.join(TEST_DIR, "punchpipe_config.yaml")
    pipeline_config = load_pipeline_configuration(pipeline_config_path)
    input_files = [File(level='Q',
                       file_type='CQ',
                       observatory='M',
                       state='created',
                       file_version='none',
                       software_version='none',
                       date_obs=datetime.now(UTC))]
    input_files[0].f_corona_models = input_files[0], input_files[0]
    output_file = levelq_CTM_construct_file_info(input_files, pipeline_config)
    flow_info = levelq_CTM_construct_flow_info(input_files, output_file[0], pipeline_config)

    assert flow_info.flow_type == 'levelq_CTM'
    assert flow_info.state == "planned"
    assert flow_info.flow_level == "Q"
    assert flow_info.priority == 1000
