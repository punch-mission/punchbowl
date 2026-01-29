import os
import sys
import hashlib
import multiprocessing
from glob import glob
from datetime import datetime
from collections import defaultdict

from astropy.io import fits
from dateutil.parser import parse as parse_datetime_str
from prefect_sqlalchemy import SqlAlchemyConnector
from punchpipe.control.db import File
from sqlalchemy.orm import Session
from tqdm import tqdm

from punchbowl.limits import LimitSet

n_procs = int(sys.argv[-1])

limits_path = '/home/samuel.vankooten/outlier_limits'
limit_files = os.listdir(limits_path)
limit_files = [os.path.join(limits_path, lf) for lf in limit_files]

outlier_limits = {}
if limit_files is not None:
    for limit_file in limit_files:
        if limit_file is None:
            continue
        limits = LimitSet.from_file(limit_file)
        file_name = os.path.basename(limit_file)
        code = file_name.split("_")[2][:2]
        obs = file_name.split("_")[2][-1]
        outlier_limits[code[1] + obs] = limits

print(f"Loaded {len(outlier_limits)} limit files")

input_files = glob('/d0/soc/data/0/*/**/PUNCH_L0_*.fits', recursive=True)
# input_files = glob('/mnt/archive/soc/data/0/*/**/PUNCH_L0_*.fits', recursive=True)

# If there are multiple versions, let's pick out the latest

versions_by_code_and_timestamp = defaultdict(list)
for path in input_files:
    fname = os.path.basename(path)
    pieces = fname.split('_')
    version = pieces[-1].split('.')[0]
    timestamp = pieces[-2]
    code = pieces[-3]
    if code[:2] == 'PX':
        continue
    if code[0] not in ("C", "P"):
        continue
    versions_by_code_and_timestamp[(timestamp, code)].append((version, path))

n_already = 0
files_to_upgrade = []
for timestamp, code in sorted(versions_by_code_and_timestamp.keys(), reverse=True):
    items = versions_by_code_and_timestamp[(timestamp, code)]
    version, path = sorted(items)[-1]
    if version == 'v0e':
        n_already += 1
        continue
    files_to_upgrade.append(path)

print(f"Identified {len(files_to_upgrade)} files to upgrade; skipping {n_already} already upgraded")

def write_file_hash(path: str) -> None:
    """Create a SHA-256 hash for a file."""
    file_hash = hashlib.sha256()
    with open(path, "rb") as f:
        fb = f.read()
        file_hash.update(fb)

    with open(path + ".sha", "w") as f:
        f.write(file_hash.hexdigest())

def upgrade_file(path):
    pieces = path.split('_')
    assert pieces[-1].endswith('.fits')
    assert pieces[-1].startswith('v0')
    source_version = pieces[-1].split('.')[0]
    pieces[-1] = 'v0e.fits'
    new_path = '_'.join(pieces)

    base_path = os.path.basename(new_path)
    code = base_path.split("_")[2][:2]
    obs = base_path.split("_")[2][-1]
    with fits.open(path, lazy_load_hdus=False, disable_image_compression=True) as hdul, Session(engine) as session:
        header = hdul[1].header
        limits = outlier_limits[header['TYPECODE'][1] + header['OBSCODE']]
        is_outlier = not limits.is_good(header)
        header.insert('BUNIT', ('OUTLIER', int(is_outlier), 'Probable bad-image status'))
        now = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
        header['DATE'] = now
        header.append(('HISTORY', f'{now[:-4]} => Upgraded from {source_version} to v0e|'))

        l0_db_entry = File(level="0",
                           polarization='C' if code[0] == 'C' else code[1],
                           file_type=code,
                           observatory=obs,
                           file_version='0e',
                           software_version='upgrade script',
                           outlier=is_outlier,
                           date_created=datetime.now(),
                           date_obs=parse_datetime_str(header['DATE-OBS']),
                           date_beg=parse_datetime_str(header['DATE-BEG']),
                           date_end=parse_datetime_str(header['DATE-END']),
                           state="created")
        session.add(l0_db_entry)
        session.commit()

        hdul.writeto(new_path, checksum=True)
    write_file_hash(new_path)


credentials = SqlAlchemyConnector.load("mariadb-creds", _sync=True)
engine = credentials.get_engine()

def initializer():
    """ensure the parent proc's database connections are not touched
    in the new connection pool"""
    engine.dispose(close=False)

# files_to_upgrade = files_to_upgrade[:1000]
with multiprocessing.Pool(n_procs, initializer=initializer) as p, tqdm(total=len(files_to_upgrade)) as pbar:
    for _ in p.imap_unordered(upgrade_file, files_to_upgrade):
        pbar.update(1)
