import os
import asyncio
from datetime import datetime, timedelta

import pytest
import yaml
from prefect.logging import disable_run_logger
from prefect.testing.utilities import prefect_test_harness
from pytest_mock_resources import create_mysql_fixture

from punchbowl.pipe.control.cleaner import cleaner
from punchbowl.pipe.control.db import Base, File, FileRelationship, Flow
from punchbowl.pipe.control.util import load_pipeline_configuration

loop: asyncio.AbstractEventLoop


@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    with prefect_test_harness():
        yield


def session_fn(session):
    first_L1_flow = Flow(flow_level="1",
                         flow_type="level1",
                         state="completed",
                         creation_time=datetime.now(),
                         priority=1)

    first_L1_flowA = Flow(flow_level="1",
                          flow_type="level1",
                          state="completed",
                          creation_time=datetime.now(),
                          priority=1)

    first_LQ_flow = Flow(flow_level="Q",
                         flow_type="levelQ",
                         state="completed",
                         creation_time=datetime.now(),
                         priority=1)

    first_L2_flow = Flow(flow_level="2",
                         flow_type="level2",
                         state="completed",
                         creation_time=datetime.now(),
                         priority=1)

    second_L1_flow = Flow(flow_level="1",
                          flow_type="level1",
                          state="completed",
                          creation_time=datetime.now(),
                          priority=1)

    second_LQ_flow = Flow(flow_level="Q",
                          flow_type="levelQ",
                          state="completed",
                          creation_time=datetime.now(),
                          priority=1)

    session.add(first_L1_flow)
    session.add(first_L1_flowA)
    session.add(first_LQ_flow)
    session.add(first_L2_flow)
    session.add(second_L1_flow)
    session.add(second_LQ_flow)
    session.commit()

    l0_file = File(level="0",
                   file_type='PM',
                   observatory='4',
                   state='progressed',
                   file_version='none',
                   software_version='none',
                   date_obs=datetime.now())

    l1_file = File(level="1",
                   file_type="PM",
                   observatory='4',
                   state='progressed',
                   file_version='none',
                   software_version='none',
                   processing_flow=first_L1_flow.flow_id,
                   date_obs=datetime.now())

    l0_fileA = File(level="0",
                    file_type='PM',
                    observatory='3',
                    state='progressed',
                    file_version='none',
                    software_version='none',
                    date_obs=datetime.now())

    l1_fileA = File(level="1",
                    file_type="PM",
                    observatory='3',
                    state='progressed',
                    file_version='none',
                    software_version='none',
                    processing_flow=first_L1_flowA.flow_id,
                    date_obs=datetime.now())

    lq_file = File(level="Q",
                   file_type="PM",
                   observatory='4',
                   state='created',
                   file_version='none',
                   software_version='none',
                   processing_flow=first_LQ_flow.flow_id,
                   date_obs=datetime.now())

    l2_file = File(level="2",
                   file_type="PM",
                   observatory='4',
                   state='created',
                   file_version='none',
                   software_version='none',
                   processing_flow=first_L2_flow.flow_id,
                   date_obs=datetime.now())

    second_l0_file = File(level="0",
                          file_type='PM',
                          observatory='4',
                          state='progressed',
                          file_version='none',
                          software_version='none',
                          date_obs=datetime.now() + timedelta(days=1))

    second_l1_file = File(level="1",
                          file_type="PM",
                          observatory='4',
                          state='quickpunched',
                          file_version='none',
                          software_version='none',
                          processing_flow=second_L1_flow.flow_id,
                          date_obs=datetime.now() + timedelta(days=1))

    second_lq_file = File(level="Q",
                          file_type="PM",
                          observatory='4',
                          state='created',
                          file_version='none',
                          software_version='none',
                          processing_flow=second_LQ_flow.flow_id,
                          date_obs=datetime.now() + timedelta(days=1))

    session.add(l0_file)
    session.add(l1_file)
    session.add(l0_fileA)
    session.add(l1_fileA)
    session.add(lq_file)
    session.add(l2_file)
    session.add(second_l0_file)
    session.add(second_l1_file)
    session.add(second_lq_file)
    session.commit()

    session.add(FileRelationship(parent=l0_file.file_id, child=l1_file.file_id))
    session.add(FileRelationship(parent=l0_fileA.file_id, child=l1_fileA.file_id))
    session.add(FileRelationship(parent=l1_file.file_id, child=lq_file.file_id))
    session.add(FileRelationship(parent=l1_fileA.file_id, child=lq_file.file_id))
    session.add(FileRelationship(parent=l1_file.file_id, child=l2_file.file_id))
    session.add(FileRelationship(parent=l1_fileA.file_id, child=l2_file.file_id))
    session.add(FileRelationship(parent=second_l0_file.file_id, child=second_l1_file.file_id))
    session.add(FileRelationship(parent=second_l1_file.file_id, child=second_lq_file.file_id))


db = create_mysql_fixture(Base, session_fn, session=True)
TEST_DIR = os.path.dirname(__file__)


@pytest.fixture
def populated_tmpdir_config(tmpdir, db):
    files = db.query(File).all()
    for file in files:
        target = os.path.join(file.directory(tmpdir), file.filename())
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, 'w'):
            pass
        with open(target + '.sha', 'w'):
            pass
        with open(target.replace('.fits.', '.jp2'), 'w'):
            pass
    config = load_pipeline_configuration(os.path.join(TEST_DIR, "punchpipe_config.yaml"))
    config['root'] = str(tmpdir)
    new_config = os.path.join(tmpdir, 'config.yaml')
    with open(new_config, 'w') as f:
        yaml.dump(config, f)
    return new_config

@pytest.mark.asyncio(loop_scope="module")
async def test_reset_L1(db, tmpdir, populated_tmpdir_config):
    global loop
    l1s = db.query(File).filter(File.level == "1").all()
    reset_file = l1s[1]
    reset_file_path = os.path.join(reset_file.directory(tmpdir), reset_file.filename())

    reset_flow = db.query(Flow).filter(Flow.flow_id == reset_file.processing_flow).one()
    reset_flow.state = 'revivable'
    db.commit()

    parent_file = (db.query(File)
                     .join(FileRelationship, File.file_id == FileRelationship.parent)
                     .filter(FileRelationship.child == reset_file.file_id)).one()
    other_files = db.query(File).filter(File.file_id != reset_file.file_id).all()
    other_flows = db.query(Flow).filter(Flow.flow_id != reset_flow.flow_id).all()

    other_file_ids = [f.file_id for f in other_files]
    other_flow_ids = [f.flow_id for f in other_flows]
    other_file_paths = [os.path.join(f.directory(tmpdir), f.filename()) for f in other_files]

    with disable_run_logger():
        await cleaner.fn(populated_tmpdir_config, session=db)

    remaining_files = db.query(File).filter(File.file_id != reset_file.file_id).all()
    remaining_flows = db.query(Flow).filter(Flow.flow_id != reset_flow.flow_id).all()

    assert [f.file_id for f in remaining_files] == other_file_ids
    assert [f.flow_id for f in remaining_flows] == other_flow_ids

    for path in other_file_paths:
        assert os.path.exists(path)
    assert not os.path.exists(reset_file_path)
    assert not os.path.exists(reset_file_path + '.sha')
    assert not os.path.exists(reset_file_path.replace('.fits.', '.jp2'))
    assert not os.path.exists(os.path.dirname(reset_file_path))

    for file in remaining_files:
        if file.file_id == parent_file.file_id:
            assert file.state == 'created'

    relationships = db.query(FileRelationship).filter(FileRelationship.child == reset_file.file_id).all()
    assert len(relationships) == 0

@pytest.mark.asyncio(loop_scope="module")
async def test_reset_LQ(db, tmpdir, populated_tmpdir_config):
    global loop

    lqs = db.query(File).filter(File.level == "Q").all()
    for reset_file in lqs:
        parent_files = (db.query(File)
                        .join(FileRelationship, File.file_id == FileRelationship.parent)
                        .filter(FileRelationship.child == reset_file.file_id)).all()
        parent_file_ids = [f.file_id for f in parent_files]
        if len(parent_file_ids) == 2:
            break
    else:
        raise RuntimeError("Didn't find the right child file!")

    reset_file_path = os.path.join(reset_file.directory(tmpdir), reset_file.filename())

    reset_flow = db.query(Flow).filter(Flow.flow_id == reset_file.processing_flow).one()
    reset_flow.state = 'revivable'
    db.commit()

    other_files = db.query(File).filter(File.file_id != reset_file.file_id).all()
    other_flows = db.query(Flow).filter(Flow.flow_id != reset_flow.flow_id).all()

    other_file_ids = [f.file_id for f in other_files]
    other_flow_ids = [f.flow_id for f in other_flows]
    other_file_paths = [os.path.join(f.directory(tmpdir), f.filename()) for f in other_files]

    with disable_run_logger():
        await cleaner.fn(populated_tmpdir_config, session=db)

    remaining_files = db.query(File).filter(File.file_id != reset_file.file_id).all()
    remaining_flows = db.query(Flow).filter(Flow.flow_id != reset_flow.flow_id).all()

    assert [f.file_id for f in remaining_files] == other_file_ids
    assert [f.flow_id for f in remaining_flows] == other_flow_ids

    for path in other_file_paths:
        assert os.path.exists(path)
    assert not os.path.exists(reset_file_path)
    assert not os.path.exists(reset_file_path + '.sha')
    assert not os.path.exists(reset_file_path.replace('.fits.', '.jp2'))
    assert not os.path.exists(os.path.dirname(reset_file_path))

    for file in remaining_files:
        if file.file_id in parent_file_ids:
            assert file.state == 'created'

    relationships = db.query(FileRelationship).filter(FileRelationship.child == reset_file.file_id).all()
    assert len(relationships) == 0

@pytest.mark.asyncio(loop_scope="module")
async def test_reset_L2(db, tmpdir, populated_tmpdir_config):
    l2s = db.query(File).filter(File.level == "2").all()
    reset_file = l2s[0]
    reset_file_path = os.path.join(reset_file.directory(tmpdir), reset_file.filename())

    reset_flow = db.query(Flow).filter(Flow.flow_id == reset_file.processing_flow).one()
    reset_flow.state = 'revivable'
    db.commit()

    parent_files = (db.query(File)
                      .join(FileRelationship, File.file_id == FileRelationship.parent)
                      .filter(FileRelationship.child == reset_file.file_id)).all()
    parent_file_ids = [f.file_id for f in parent_files]
    assert len(parent_file_ids) == 2
    other_files = db.query(File).filter(File.file_id != reset_file.file_id).all()
    other_flows = db.query(Flow).filter(Flow.flow_id != reset_flow.flow_id).all()

    other_file_ids = [f.file_id for f in other_files]
    other_flow_ids = [f.flow_id for f in other_flows]
    other_file_paths = [os.path.join(f.directory(tmpdir), f.filename()) for f in other_files]

    with disable_run_logger():
        await cleaner.fn(populated_tmpdir_config, session=db)

    remaining_files = db.query(File).filter(File.file_id != reset_file.file_id).all()
    remaining_flows = db.query(Flow).filter(Flow.flow_id != reset_flow.flow_id).all()

    assert [f.file_id for f in remaining_files] == other_file_ids
    assert [f.flow_id for f in remaining_flows] == other_flow_ids

    for path in other_file_paths:
        assert os.path.exists(path)
    assert not os.path.exists(reset_file_path)
    assert not os.path.exists(reset_file_path + '.sha')
    assert not os.path.exists(reset_file_path.replace('.fits.', '.jp2'))
    assert not os.path.exists(os.path.dirname(reset_file_path))

    for file in remaining_files:
        if file.file_id in parent_file_ids:
            assert file.state == 'created'

    relationships = db.query(FileRelationship).filter(FileRelationship.child == reset_file.file_id).all()
    assert len(relationships) == 0
