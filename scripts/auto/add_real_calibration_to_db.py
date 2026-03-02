import os
import sys
import shutil
from glob import glob
from datetime import datetime

from punchbowl.auto.control.db import File
from punchbowl.auto.control.util import get_database_session

model_directory = sys.argv[1]
root_dir = sys.argv[2]

session = get_database_session()

model_paths = sorted(glob(os.path.join(model_directory, "PUNCH_L1_R*.fits")))
model_paths += sorted(glob(os.path.join(model_directory, "PUNCH_L1_FQ*.fits")))
model_paths += sorted(glob(os.path.join(model_directory, "PUNCH_L1_G*.fits")))
model_paths += sorted(glob(os.path.join(model_directory, "PUNCH_L1_S*.fits")))
model_paths += sorted(glob(os.path.join(model_directory, "PUNCH_L1_D*.fits")))
model_paths += sorted(glob(os.path.join(model_directory, "PUNCH_L1_MS*.bin")))
model_paths += sorted(glob(os.path.join(model_directory, "PUNCH_L0_L*.npz")))

for model_path in model_paths:
    base_path = os.path.basename(model_path)
    level = base_path.split("_")[1][1]
    code = base_path.split("_")[2][:2]
    obs = base_path.split("_")[2][-1]
    date = datetime.strptime(base_path.split("_")[3], "%Y%m%d%H%M%S")
    version = base_path.split("_")[-1].split(".")[0][1:]

    file = File(
        level=level,
        file_type=code,
        observatory=obs,
        file_version=version,
        software_version=version,
        date_obs=date,
        polarization=code[1] if code[0] == 'P' else 'C',
        outlier=0,
        state='created',
    )
    session.add(file)
    output_filename = os.path.join(
        file.directory(root_dir), file.filename()
    )
    if code == "MS":
        output_filename = output_filename.replace(".fits", ".bin")
    if code[0] == "L":
        output_filename = output_filename.replace(".fits", ".npz")
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    shutil.copyfile(model_path, output_filename)
    print(f"Created {output_filename}")
session.commit()
