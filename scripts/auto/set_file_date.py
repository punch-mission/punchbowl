import os
import sys
import shutil
from glob import glob
from pathlib import Path
from datetime import datetime

from astropy.io import fits

target = sys.argv[1]
new_date = sys.argv[2]

# We have a directory tree---find the (hopefully) single file in it
if not target.endswith('.fits') and not target.endswith('.bin'):
    candidates = glob(f"{target}/**/*.fits", recursive=True)
    if len(candidates) == 0:
        candidates = glob(f"{target}/**/*.bin", recursive=True)
    if len(candidates) != 1:
        # We can't handle multiple files right now
        print(f"For target {target}, found these possible files:")
        for c in candidates:
            print(f"  {c}")
        if not len(candidates):
            print("  [none]")
        sys.exit()
    target = candidates[0]

target = Path(target)

# Parse the new date
for format in ["%Y%m%d%H%M%S", "%Y%m%dT%H%M%S", "%Y-%m-%d %H:%M:%S"]:
    try:
        date = datetime.strptime(new_date, format)
        break
    except ValueError:
        pass
else:
    raise ValueError("Couldn't parse date")

# Parse the file name and change the date
name = target.name
pieces = name.split('_')
pieces[3] = date.strftime("%Y%m%d%H%M%S")


output_path = list(target.parts)
output_path[-1] = '_'.join(pieces)

# If this file is in a YYYY/MM/DD directory structure, we'll move it
if len(output_path) >= 4:
    path_has_date = len(output_path[-4]) == 4 and len(output_path[-3]) == 2 and len(output_path[-2]) == 2
    for piece in output_path[-4:-1]:
        try:
            int(piece)
        except ValueError:
            path_has_date = False
    if path_has_date:
        output_path[-2] = date.strftime("%d")
        output_path[-3] = date.strftime("%m")
        output_path[-4] = date.strftime("%Y")
output_path = Path(*output_path)

# print(target, new_date, date, output_path)

os.makedirs(output_path.parent, exist_ok=True)


if target.name.endswith('.fits'):
    # Update FITS metadata and copy the file
    with fits.open(target) as hdul:
        hdul[1].header['DATE-OBS'] = date.strftime("%Y-%m-%dT%H:%M:%S.%f")
        hdul.writeto(output_path)
else:
    shutil.copyfile(target, output_path)

target.unlink()

for parent_dir in target.parents:
    if not parent_dir.exists():
        break
    if len(os.listdir(parent_dir)):
        break
    if parent_dir == Path.cwd():
        break
    parent_dir.rmdir()
