import os
from datetime import datetime

import pandas as pd
import pytest

from punchbowl.auto.cli import clean_replay

TEST_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(TEST_DIR, "punchpipe_config.yaml")

def test_clean_replay():
    input_file = os.path.join(TEST_DIR, "data/test_replay.csv")
    result = clean_replay(input_file, CONFIG_PATH, write=False, reference_date=datetime(2025,6,6))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 34


def test_clean_replay_notime():
    input_file = os.path.join(TEST_DIR, "data/test_replay.csv")
    result = clean_replay(input_file, CONFIG_PATH, write=False, window_in_days=None, reference_date=datetime(2025,6,6))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 34


def test_clean_replay_empty():
    input_file = os.path.join(TEST_DIR, "data/test_replay_empty.csv")
    result = clean_replay(input_file, CONFIG_PATH, write=False, reference_date=datetime(2025,6,6))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_clean_replay_one_request():
    input_file = os.path.join(TEST_DIR, "data/test_replay_one.csv")
    result = clean_replay(input_file, CONFIG_PATH, write=False, reference_date=datetime(2025,6,6))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1


def test_clean_replay_connected_requests():
    input_file = os.path.join(TEST_DIR, "data/test_replay_connected.csv")
    result = clean_replay(input_file, CONFIG_PATH, write=False, reference_date=datetime(2025,5,29))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
