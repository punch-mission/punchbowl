import os
import sys
import hashlib
import multiprocessing
from glob import glob
from datetime import UTC, datetime

from astropy.io import fits
from dateutil.parser import parse as parse_datetime_str
from prefect_sqlalchemy import SqlAlchemyConnector
from sqlalchemy.orm import Session
from tqdm import tqdm

from punchbowl.auto.control.db import File
from punchbowl.limits import LimitSet

n_procs = int(sys.argv[-1])

base_dir = '/d0/soc/data/'

limits_path = '/home/samuel.vankooten/outlier_limits/v0l'
limit_files = glob(os.path.join(limits_path, '*.npz'))

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

print(f"Loaded {len(outlier_limits)} limit files")

credentials = SqlAlchemyConnector.load("mariadb-creds", _sync=True)
engine = credentials.get_engine()
session = Session(engine)

target_files = (session.query(File).where(File.level == '0')
                .where(File.observatory != '4').all())
target_paths = [os.path.join(f.directory('/d0/punchsoc/real_data/'), f.filename()) for f in target_files]

print(f"Identified {len(target_files)} files to re-write")

def write_file_hash(path: str) -> None:
    """Create a SHA-256 hash for a file."""
    file_hash = hashlib.sha256()
    with open(path, "rb") as f:
        fb = f.read()
        file_hash.update(fb)

    with open(path + ".sha", "w") as f:
        f.write(file_hash.hexdigest())

def rewrite_file(path):
    try:
        with fits.open(path, lazy_load_hdus=False, disable_image_compression=True, mode='update') as hdul:
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
                header['HISTORY'] = (f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%S} => rewrite-outlier-flags => Outlier "
                                     f"detection re-done with {limit_filename}|")
                break
            if selected_limits is None:
                raise RuntimeError(f"Could not find outlier limits for {path}")
            else:
                is_outlier = not selected_limits.is_good(header)

            is_outlier = is_outlier or header['BADPKTS']

            header['OUTLIER'] = int(is_outlier)
            now = datetime.now(UTC)
            header['DATE'] = now.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]

            hdul.flush()
        write_file_hash(path)
        return is_outlier
    except:
        print(f"Error loading {path}")
        raise

with multiprocessing.Pool(n_procs) as p, tqdm(total=len(target_files)) as pbar:
    for i, (f, new_outlier_status) in enumerate(zip(target_files, p.imap(rewrite_file, target_paths, chunksize=5))):
        f.outlier = new_outlier_status
        pbar.update(1)
        if i % 4000 == 0:
            session.commit()
session.commit()
