"""This is for if you're keeping a mirror of the L0 files on punch190 (to process to L1 with newer code). This script
will add the files in the given directories to the database, skipping files that are already in the database."""
import os
import sys
import multiprocessing
from glob import glob
from datetime import datetime

from astropy.io import fits
from sqlalchemy.orm import load_only
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from punchbowl.auto.control.db import File
from punchbowl.auto.control.util import get_database_session


def read_new_file_metadata(file, path):
    # Get the correct number of microseconds from the FITS header
    if file.file_type[0] in ['M', 'L']:
        return None, None, None, None, None

    new_dateobs, new_date_created, new_outlier, new_bad_packets, new_crota = None, None, None, None, None
    try:
        with fits.open(path, disable_image_compression=True) as hdul:
            if len(hdul) > 1 and 'DATE-OBS' in hdul[1].header:
                p = hdul[1].header['DATE-OBS'].split('.')
                if len(p) == 2:
                    ms = p[1]
                    ms = ms + '0' * (6 - len(ms))
                    new_dateobs = file.date_obs.replace(microsecond=int(ms))
            if len(hdul) > 1 and 'DATE' in hdul[1].header:
                date = hdul[1].header['DATE']
                p = date.split('.')
                if len(p) == 2:
                    new_date_created = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%f")
                elif date:
                    new_date_created = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S")

            if len(hdul) > 1 and 'OUTLIER' in hdul[1].header:
                new_outlier = hdul[1].header['OUTLIER']

            if len(hdul) > 1 and 'BADPKTS' in hdul[1].header:
                new_bad_packets = hdul[1].header['BADPKTS']

            if len(hdul) > 1 and 'CROTA' in hdul[1].header:
                new_crota = hdul[1].header['CROTA']
    except:
        print(f"Error loading {path}")

    return new_dateobs, new_date_created, new_outlier, new_bad_packets

if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    root_dirs = sys.argv[1:]

    session = get_database_session()

    files = set()
    for root_dir in root_dirs:
        files.update(glob(f"{root_dir}/**/*.fits", recursive=True))
        files.update(glob(f"{root_dir}/**/*.bin", recursive=True))
        files.update(glob(f"{root_dir}/**/*.npz", recursive=True))

    print(f"Found {len(files)} files on disk")

    observed_levels = set()
    for path in files:
        base_path = os.path.basename(path)
        level = base_path.split("_")[1][1]
        observed_levels.add(level)

    existing_files = (
            session.query(File)
            .options(load_only(
            File.date_obs, File.file_type, File.observatory, File.file_version, File.level,
                 raiseload=True))
            .where(File.level.in_(observed_levels))
            .all())

    existing_files = {f.filename() for f in existing_files}

    print(f"Loaded {len(existing_files)} existing files from the DB")

    n_added = 0
    n_existing = 0
    new_files = []
    new_file_paths = []
    print("Identifying new files...")
    for path in tqdm(files):
        base_path = os.path.basename(path)
        level = base_path.split("_")[1][1]
        code = base_path.split("_")[2][:2]
        obs = base_path.split("_")[2][-1]
        version = base_path.split("_")[-1].split(".")[0][1:]
        date = datetime.strptime(base_path.split("_")[3], "%Y%m%d%H%M%S")

        pol = 'C'
        if code[0] in ['G', 'S', 'R', 'P', 'L']:
            pol = code[1]
            if pol == 'R':
                pol = 'C'
        if code[0] == 'X':
            pol = 'X'

        file = File(
            level=level,
            file_type=code,
            observatory=obs,
            file_version=version,
            software_version='imported to db',
            date_obs=date,
            polarization=pol,
            state='created',
            outlier=0,
        )

        if file.filename() not in existing_files:
            new_files.append(file)
            new_file_paths.append(path)
            n_added += 1
        else:
            n_existing += 1


    print(f"Found {n_added} new files, skipping {n_existing} existing files")

    print("Reading metadata...")
    for file, (date_obs, date_created, new_outlier, new_bad_packets, new_crota) in zip(
            new_files, process_map(read_new_file_metadata, new_files, new_file_paths, chunksize=10, max_workers=10)):
        if date_obs is not None:
            file.date_obs = date_obs
        if date_created is not None:
            file.date_created = date_created
        if new_outlier is not None:
            file.outlier = new_outlier
        if new_bad_packets is not None:
            file.bad_packets = new_bad_packets
        if new_crota is not None:
            file.crota = new_crota

    print("Adding to DB...")
    session.bulk_save_objects(new_files)
    session.commit()
