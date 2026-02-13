import os

import pandas as pd
from sqlalchemy import text

from punchbowl.auto.control.util import get_database_session

session, engine = get_database_session(get_engine=True)

bad_timestamps = session.execute(
    text('select distinct(timestamp) from SCI_XFI where is_used = 0;')
)

print(bad_timestamps)

table = []
for bad_timestamp in bad_timestamps:
    print(bad_timestamp)
    reason = session.execute(text(f'select distinct(last_skip_reason) '
                                  f'from SCI_XFI where timestamp = "{bad_timestamp[0].isoformat()}";')).first()[0]
    compression = session.execute(text(f'select distinct(SCI_XFI_HDR_COM_SET) '
                                  f'from SCI_XFI where timestamp = "{bad_timestamp[0].isoformat()}";')).first()[0]
    table.append({'timestamp': bad_timestamp[0], 'reason': reason, "compression_settings": compression})

df = pd.DataFrame(table)
df.to_csv('bad_image_table.csv', index=False)

bad_tlm_files = session.execute(text("select path from tlm_files where successful=0")).all()
print(bad_tlm_files)

df2 = pd.DataFrame([{"tlm": os.path.basename(path[0])} for path in bad_tlm_files])
df2.to_csv('bad_tlm_files.csv', index=False)
