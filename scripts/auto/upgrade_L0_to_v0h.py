import os
import sys
import hashlib
import multiprocessing
from glob import glob
from datetime import UTC, datetime
from collections import defaultdict

import numpy as np
from astropy.io import fits
from dateutil.parser import parse as parse_datetime_str
from prefect_sqlalchemy import SqlAlchemyConnector
from sqlalchemy.orm import Session
from tqdm import tqdm

from punchbowl.auto.control.db import File
from punchbowl.limits import LimitSet
from punchbowl.util import load_mask_file

n_procs = int(sys.argv[-1])

base_dir = '/d0/soc/data/'

limits_path = f'{base_dir}/0/L*'
limit_files = glob(os.path.join(limits_path, '**/*.npz'), recursive=True)

masks_path = f'{base_dir}/1/M*'
mask_files = glob(os.path.join(masks_path, '**/*.bin'), recursive=True)

outlier_limits = []
if limit_files is not None:
    for limit_file in limit_files:
        limits = LimitSet.from_file(limit_file)
        file_name = os.path.basename(limit_file)
        code = file_name.split("_")[2][1]
        obs = file_name.split("_")[2][2]
        version = file_name.split("_")[-1].split('.')[0]
        date = datetime.strptime(file_name.split('_')[3], '%Y%m%d%H%M%S')
        outlier_limits.append((obs, code, date, file_name, limits, version))

outlier_limits.sort(key=lambda x: (x[-1], x[2]), reverse=True)

masks = []
if mask_files is not None:
    for mask_file in mask_files:
        mask = load_mask_file(mask_file)
        filename = os.path.basename(mask_file)
        observatory = filename.split('_')[2][2]
        date = datetime.strptime(filename.split('_')[3], '%Y%m%d%H%M%S')
        masks.append((observatory, date, mask))

outlier_limits.sort(key=lambda x: (x[1]), reverse=True)

print(f"Loaded {len(outlier_limits)} limit files")
print(f"Loaded {len(masks)} mask files")

input_files = glob(f'{base_dir}/0/[CP][RMZP]*/**/PUNCH_L0_*.fits', recursive=True)

# If there are multiple versions, let's pick out the latest

versions_by_code_and_timestamp = defaultdict(list)
for path in input_files:
    fname = os.path.basename(path)
    pieces = fname.split('_')
    version = pieces[-1].split('.')[0]
    timestamp = parse_datetime_str(pieces[-2])
    code = pieces[-3]
    if code[:2] == 'PX':
        continue
    versions_by_code_and_timestamp[(timestamp, code)].append((version, path))

n_already = 0
files_to_upgrade = []
for timestamp, code in sorted(versions_by_code_and_timestamp.keys(), reverse=True):
    items = versions_by_code_and_timestamp[(timestamp, code)]
    version, path = sorted(items)[-1]
    if version == 'v0h':
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
    pieces[-1] = 'v0h.fits'
    new_path = '_'.join(pieces)

    base_path = os.path.basename(new_path)
    code = base_path.split("_")[2][:2]
    obs = base_path.split("_")[2][-1]
    try:
        with fits.open(path, lazy_load_hdus=False) as hdul, Session(engine) as session:
            header = hdul[1].header
            selected_limits = None
            for limit_observatory, limit_type, limit_date, limit_filename, limits, version in outlier_limits:
                if limit_observatory != header['OBSCODE']:
                    continue
                if limit_type != header['TYPECODE'][1]:
                    continue
                if limit_date > parse_datetime_str(header['DATE-OBS']):
                    continue
                selected_limits = limits
                header['HISTORY'] = (f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%S} => upgrade_L0_to_v0h => Outlier detection "
                                     f"with {limit_filename}|")
                break
            if selected_limits is None:
                raise RuntimeError(f"Could not find outlier limits for {path}")
            else:
                is_outlier = not selected_limits.is_good(header)

            selected_mask = None
            for mask_observatory, mask_date, mask in masks:
                if mask_observatory != header['OBSCODE']:
                    continue
                if mask_date > parse_datetime_str(header['DATE-OBS']):
                    continue
                selected_mask = mask
                break
            if selected_mask is None:
                raise RuntimeError(f"Could not find mask for {path}")
            else:
                bad_packets = np.any(hdul[1].data[~selected_mask])

            is_outlier = is_outlier or bad_packets

            header['OUTLIER'] = int(is_outlier)
            header.insert('OUTLIER', ('BADPKTS', int(bad_packets), 'Whether image data suffers from corrupted packets'))
            now = datetime.now(UTC)
            header['DATE'] = now.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
            header['HISTORY'] = f'{now:%Y-%m-%dT%H:%M:%S} => upgrade_L0_to_v0h => Upgraded from {source_version} to v0h|'

            if 'GAINLEFT' in header:
                header.insert('GAINLEFT', ('GAINBTM', header['GAINLEFT'], '[e/DN] gain value (bottom CCD side)'))
                del header['GAINLEFT']
            if 'GAINRGHT' in header:
                header.insert('GAINRGHT', ('GAINTOP', header['GAINRGHT'], '[e/DN] gain value (top CCD side)'))
                del header['GAINRGHT']
            if 'GAINCMDL' in header:
                header.insert('GAINCMDL', ('GAINCMDB', header['GAINCMDL'], '[e/DN] commanded gain value (bottom CCD side)'))
                del header['GAINCMDL']
            if 'GAINCMDR' in header:
                header.insert('GAINCMDR', ('GAINCMDT', header['GAINCMDR'], '[e/DN] commanded gain value (top CCD side)'))
                del header['GAINCMDR']

            header['FILEVRSN'] = '0h'

            hdul.writeto(new_path, checksum=True)

            l0_db_entry = File(level="0",
                               polarization='C' if code[0] == 'C' else code[1],
                               file_type=code,
                               observatory=obs,
                               file_version=header['FILEVRSN'],
                               software_version='upgrade script',
                               outlier=is_outlier,
                               bad_packets=bad_packets,
                               date_created=now.astimezone(),
                               date_obs=parse_datetime_str(header['DATE-OBS']),
                               date_beg=parse_datetime_str(header['DATE-BEG']),
                               date_end=parse_datetime_str(header['DATE-END']),
                               state="upgraded")
            session.add(l0_db_entry)
            session.commit()
        write_file_hash(new_path)
    except:
        print(f"Error loading {path}")
        raise


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
