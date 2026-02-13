import os
import json
from datetime import datetime, timedelta
from functools import cache
from collections import defaultdict

from punchbowl.auto.control.db import Flow
from punchbowl.auto.control.util import get_database_session

session = get_database_session()

for state in ['running', 'failed', 'launched']:
    interrupted_flows = session.query(Flow).where(Flow.state == state).where(Flow.flow_level == "S").all()
    print()
    print(f"There are {len(interrupted_flows)} flows marked as '{state}'.")
    if len(interrupted_flows) and input("Change them to 'planned'? y/N: ").lower() == 'y':
        for flow in interrupted_flows:
            flow.state = 'planned'
            flow.flow_run_name = None
            flow.flow_run_id = None
            flow.start_time = None
        session.commit()

planned_flows = session.query(Flow).where(Flow.state == "planned").where(Flow.flow_level == "S").all()

@cache
def get_files_in_dir_by_timestamp(dir):
    all_files = []
    for path, _, files in os.walk(dir):
        for file in files:
            all_files.append(os.path.join(path, file))
    files_by_timestamp = defaultdict(list)
    for file in all_files:
        base = os.path.basename(file)
        _, level, code, timestamp, *_ = base.split('_')
        files_by_timestamp[timestamp].append(file)
    return files_by_timestamp

target_files = []
roots = set()
delta = timedelta(minutes=4)
for flow in planned_flows:
    data = json.loads(flow.call_data)
    date = datetime.fromisoformat(data['date_obs'])
    date += delta / 2
    timestamp = date.strftime("%Y%m%d%H%M%S")
    files_by_timestamp = get_files_in_dir_by_timestamp(data['out_dir'])
    files = files_by_timestamp[timestamp]
    if len(files):
        roots.add(data['out_dir'])
        target_files.extend(files)

print(f"\nThere are {len(target_files)} existing files for flows marked as 'planned'.")
print("They are under these top-level directories:")
for root in roots:
    print(f"   - {root}")
if len(target_files) and input('Delete them? y/N: ').lower() == 'y':
    for file in target_files:
        os.remove(file)
