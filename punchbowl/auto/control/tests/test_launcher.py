import os
import math
from datetime import UTC, datetime

import pytest
from freezegun import freeze_time
from prefect.logging import disable_run_logger
from prefect.testing.utilities import prefect_test_harness
from pytest_mock_resources import create_mysql_fixture

from punchbowl.auto.control.db import Base, File, Flow
from punchbowl.auto.control.launcher import (
    count_flows,
    determine_launchable_flow_count,
    escalate_long_waiting_flows,
    gather_planned_flows,
)
from punchbowl.auto.control.util import load_pipeline_configuration

TEST_DIR = os.path.dirname(__file__)


def session_fn(session):
    level0_file = File(level=0,
                       file_type='XX',
                       observatory='0',
                       state='created',
                       file_version='none',
                       software_version='none',
                       date_obs=datetime.now(UTC))

    level1_file = File(level=1,
                       file_type="XX",
                       observatory='0',
                       state='created',
                       file_version='none',
                       software_version='none',
                       date_obs=datetime.now(UTC))

    level0_planned_flow = Flow(flow_id=1,
                               flow_level=0,
                              flow_type='level0',
                              state='planned',
                              creation_time=datetime(2023, 2, 2, 0, 0, 0),
                              priority=5)

    level1_planned_flow = Flow(flow_id=2,
                               flow_level=1,
                              flow_type='level1',
                              state='planned',
                              creation_time=datetime(2023, 2, 2, 0, 0, 0),
                              priority=2)

    level1_planned_flow2 = Flow(flow_id=3,
                               flow_level=1,
                              flow_type='level1',
                              state='planned',
                              creation_time=datetime(2023, 2, 2, 0, 0, 0),
                              priority=100)

    session.add(level0_file)
    session.add(level1_file)
    session.add(level0_planned_flow)
    session.add(level1_planned_flow)
    session.add(level1_planned_flow2)


db = create_mysql_fixture(Base, session_fn, session=True)
db_empty = create_mysql_fixture(Base, session=True)

@pytest.fixture
def flow_weights():
    return {"level0": 1, "level1": 2}


@pytest.fixture
def flow_batch_sizes():
    return {"level0": 1, "level1": 1}


def test_gather_queued_flows(db, flow_weights, flow_batch_sizes):
    planned_ids, tags_by_flow, selected_weight, number_of_flows, count_per_type = gather_planned_flows.fn(
        db, 6, 3, flow_weights, {"level0": True, "level1": True}, flow_batch_sizes)
    assert len(planned_ids) == 3
    assert count_per_type["level0"] == 1
    assert count_per_type["level1"] == 2
    assert selected_weight == 5
    assert number_of_flows == 3
    planned_ids, tags_by_flow, selected_weight, number_of_flows, count_per_type = gather_planned_flows.fn(
        db, 3, 3, flow_weights,{"level0": True, "level1": True}, flow_batch_sizes)
    assert len(planned_ids) == 2
    assert count_per_type["level0"] == 1
    assert count_per_type["level1"] == 1
    assert selected_weight == 3
    assert number_of_flows == 2


def test_count_running_flows(db, flow_weights):
    running_count, planned_count, weight_planned, weight_running = count_flows.fn(db, flow_weights)
    assert running_count == 0
    assert planned_count == 3
    assert weight_running == 0
    assert weight_planned == 5


def test_escalate_long_waiting_flows(db):
    pipeline_config_path = os.path.join(TEST_DIR, "punchpipe_config.yaml")
    pipeline_config = load_pipeline_configuration(pipeline_config_path)

    with freeze_time(datetime(2023, 2, 2, 0, 0, 0)) as frozen_datetime:
        escalate_long_waiting_flows.fn(db, pipeline_config)
        assert db.query(Flow).where(Flow.flow_id == 1).one().priority == 5

        frozen_datetime.move_to(datetime(2023, 2, 2, 0, 0, 31))
        escalate_long_waiting_flows.fn(db, pipeline_config)
        assert db.query(Flow).where(Flow.flow_id == 1).one().priority == 10

        frozen_datetime.move_to(datetime(2024, 2, 2, 0, 0, 0))
        escalate_long_waiting_flows.fn(db, pipeline_config)
        assert db.query(Flow).where(Flow.flow_id == 1).one().priority == 30


def test_filter_for_launchable_flows(db, flow_weights):
    with prefect_test_harness(), disable_run_logger():
        running_count, planned_count, weight_planned, weight_running = count_flows.fn(db, flow_weights)
        max_weight_running = 30
        ready_to_launch_weight, max_flows_to_launch = determine_launchable_flow_count(
            weight_planned, weight_running, max_weight_running, math.inf, 10)
        assert ready_to_launch_weight == 5
        assert max_flows_to_launch == 10


def test_filter_for_launchable_flows_with_max_of_1(db, flow_weights, flow_batch_sizes):
    with prefect_test_harness(), disable_run_logger():
        running_count, planned_count, weight_planned, weight_running = count_flows.fn(db, flow_weights)
        max_weight_running = 1
        ready_to_launch_weight, max_flows_to_launch = determine_launchable_flow_count(
            weight_planned, weight_running, max_weight_running, math.inf, 10)
        assert ready_to_launch_weight == 1
        assert max_flows_to_launch == 10
        flows, tags_by_flow, selected_weight, number_of_flows, count_per_type = gather_planned_flows.fn(
            db, ready_to_launch_weight, max_flows_to_launch, flow_weights, {"level0": True, "level1": True},
            flow_batch_sizes)
        assert len(flows) == 1
        assert flows[0][0].flow_id == 3
        assert number_of_flows == 1


def test_filter_for_launchable_flows_with_max_of_0(db, flow_weights):
    with prefect_test_harness(), disable_run_logger():
        running_count, planned_count, weight_planned, weight_running = count_flows.fn(db, flow_weights)
        max_weight_running = 0
        ready_to_launch_weight, max_flows_to_launch = determine_launchable_flow_count(
            weight_planned, weight_running, max_weight_running, math.inf, 0)
        assert ready_to_launch_weight == 0
        assert max_flows_to_launch == 0


def test_filter_for_launchable_flows_with_empty_db(db_empty, flow_weights):
    with prefect_test_harness(), disable_run_logger():
        running_count, planned_count, weight_planned, weight_running = count_flows.fn(db_empty, flow_weights)
        max_weight_running = 30
        ready_to_launch_weight, max_flows_to_launch = determine_launchable_flow_count(
            weight_planned, weight_running, max_weight_running, math.inf, 20)
        assert ready_to_launch_weight == 0
        assert max_flows_to_launch == 20


def test_launch_ready_flows():
    pass
