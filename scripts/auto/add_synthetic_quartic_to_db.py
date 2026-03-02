import os
import sys
import shutil
from datetime import datetime

from punchbowl.auto.control.db import File
from punchbowl.auto.control.util import get_database_session

wfi_path = sys.argv[1]
nfi_path = sys.argv[2]
root_dir = sys.argv[3]

session = get_database_session()

code = 'FQ'
for obs in ['1', '2', '3', '4']:
    file = File(
        level="1",
        file_type=code,
        observatory=obs,
        file_version="1",
        software_version="synth",
        date_obs=datetime.fromisoformat("2000-01-01 00:00:00"),
        state='created',
    )
    session.add(file)
    output_filename = os.path.join(
        file.directory(root_dir), file.filename()
    )
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    shutil.copyfile(nfi_path if obs == '4' else wfi_path, output_filename)
    print(f"Created {output_filename}")
session.commit()
