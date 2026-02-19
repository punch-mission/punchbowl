from collections import defaultdict

from sqlalchemy.orm import aliased

from punchbowl.auto.control.db import File, FileRelationship, Flow
from punchbowl.auto.control.util import get_database_session


def mkrecord(f, fate, flow):
    d = f.filename(), f.level, f.date_obs.strftime("%Y-%m-%d %H:%M:%S.%f"), f.file_type, f.observatory, f.date_created.strftime("%Y-%m-%d %H:%M:%S.%f"), f.state, fate
    if flow is not None:
        d = d + (flow.flow_id, flow.flow_run_id, flow.flow_run_name, flow.start_time.strftime("%Y-%m-%d %H:%M:%S.%f"))
    return d

session = get_database_session()

L0s = session.query(File).filter(File.level == '0').where(File.file_type.in_(['CR', 'PZ', 'PM', 'PP'])).all()

l0_groups = defaultdict(list)

for f in L0s:
    key = (f.observatory, f.file_type, f.date_obs)
    l0_groups[key].append(f)

dup_groups = [g for g in l0_groups.values() if len(g) > 1]

child = aliased(File)
parent = aliased(File)


to_delete = []
records = []
for group in dup_groups:
    group = sorted(group, key=lambda f: f.date_created)
    duplicates = group[1:]
    to_delete.extend(duplicates)
    records.append(mkrecord(group[0], 'kept', None))
    for f in duplicates:
        records.append(mkrecord(f, 'deleted', None))
    # L1s = (session.query(File, Flow)
    #        .join(FileRelationship, FileRelationship.child == File.file_id)
    #        .join(Flow, Flow.flow_id == File.processing_flow)
    #        .filter(FileRelationship.parent.in_([f.file_id for f in duplicates]))
    #        .all())
    # for L1, flow in L1s:
    #     to_delete.append(L1)
    #     records.append(mkrecord(L1, 'deleted', flow))

with open('duplicate-records.csv', 'w') as f:
    f.write('filename_stub,level,date_obs,file_type,observatory,date_created,state,fate\n')
    for r in records:
        f.write(','.join(r) + '\n')

for record in to_delete:
    session.delete(record)
session.commit()
