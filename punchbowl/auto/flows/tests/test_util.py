import os
from datetime import datetime, timedelta

from punchbowl.auto.control.db import File
from punchbowl.auto.flows.util import apply_stride

TEST_DIR = os.path.dirname(__file__)


def test_apply_stride():
    dateobs = datetime(2026, 1, 1)
    cadence = 8
    full_cadence_files = [
        File(level="1",
             file_type='CR',
             observatory='1',
             state='created',
             file_version='none',
             software_version='none',
             date_obs=dateobs + timedelta(minutes=cadence) * i)
        for i in range(20)
    ]

    half_cadence_files = [
        File(level="1",
             file_type='CR',
             observatory='1',
             state='created',
             file_version='none',
             software_version='none',
             date_obs=full_cadence_files[-1].date_obs + timedelta(minutes=cadence) * i)
        for i in range(2, 20, 2)
    ]

    files = full_cadence_files + half_cadence_files
    selected = apply_stride(files, 2, cadence)
    assert len(selected) == len(full_cadence_files) // 2 + len(half_cadence_files)
    assert all(f in selected for f in half_cadence_files)
    assert not all(f in selected for f in full_cadence_files)
