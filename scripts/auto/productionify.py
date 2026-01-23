import os
import sys
import argparse
from itertools import repeat
from concurrent.futures.process import ProcessPoolExecutor

import numpy as np
from dateutil.parser import parse as parse_datetime_str
from sqlalchemy import or_
from tqdm.auto import tqdm

from punchbowl.auto.control.db import File
from punchbowl.auto.control.util import (
    _write_quicklook,
    get_database_session,
    load_pipeline_configuration,
    replace_file_version_in_metadata,
)
from punchbowl.data.punch_io import load_ndcube_from_fits, write_file_hash


def productionify_file(file: File, config: dict, data_root: str, old_pattern, new_version):
    try:
        old_path = os.path.join(file.directory(data_root), file.filename())
        file.file_version = new_version
        new_path = os.path.join(file.directory(data_root), file.filename())

        if os.path.exists(old_path):
            replace_file_version_in_metadata(old_path, old_pattern, new_version)
            os.rename(old_path, new_path)

        old_sha_path = old_path + '.sha'
        new_sha_path = new_path + '.sha'
        if os.path.exists(old_sha_path):
            os.rename(old_sha_path, new_sha_path)
        elif os.path.exists(new_path):
            # We should overwrite existing sha files because we might have changed the metadata
            write_file_hash(new_path)

        old_ql_path = old_path.replace('.fits', '.jp2')
        new_ql_path = new_path.replace('.fits', '.jp2')
        if os.path.exists(old_ql_path):
            os.rename(old_ql_path, new_ql_path)
        elif os.path.exists(new_path) and not os.path.exists(new_ql_path) and file.file_type[0] not in ('S', 'T'):
            cube = load_ndcube_from_fits(new_path)
            with np.errstate(all='ignore'):
                _write_quicklook(config, file, cube)
        return True
    except Exception as e:
        msg = f"Error in {file.filename()}, {repr(e)}"
        print(msg)
        return msg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--level', action='append')
    parser.add_argument('-t', '--type', action='append')
    parser.add_argument('-o', '--obs', action='append')
    parser.add_argument('-v', '--file_version', action='append')
    parser.add_argument('--dobs_start')
    parser.add_argument('--dobs_end')
    parser.add_argument('--dcreate_start')
    parser.add_argument('--dcreate_end')
    parser.add_argument('-n', '--max-n-files', type=int)
    parser.add_argument('-f', '--force', action='store_true', help="Skip warning message")
    parser.add_argument ('-w', '--workers', type=int, help="Number of worker processes", default=12)
    parser.add_argument('data_root')
    parser.add_argument('pipeline_config')
    parser.add_argument('old_version_pattern')
    parser.add_argument('new_version')

    args = parser.parse_args()

    if not args.force:
        print("This script should not run when files of the selected type are being produced, are planned (even if not "
              "running), or if the selected files have descendants that are planned.")
        print("This is because planned flows will have file names written in the call_data in the database, "
              "but we'll be renaming files.")
        input("Press enter to acknowledge this.")

    old_version_pattern = args.old_version_pattern
    if old_version_pattern.startswith('v'):
        print("Stripping leading 'v' from old version")
        old_version_pattern = old_version_pattern[1:]

    new_version = args.new_version
    if new_version.startswith('v'):
        print("Stripping leading 'v' from new version")
        new_version = new_version[1:]

    config = load_pipeline_configuration(args.pipeline_config)
    session = get_database_session(session_kwargs=dict(expire_on_commit=False))

    query = session.query(File).where(File.file_version != new_version)
    if args.level:
        query = query.where(File.level.in_(args.level))
    if args.type:
        if any('%' in t for t in args.type):
            conditions = [File.file_type.like(t) for t in args.type]
            query = query.where(or_(*conditions))
        else:
            query = query.where(File.file_type.in_(args.type))
    if args.obs:
        query = query.where(File.observatory.in_(args.obs))
    if args.file_version:
        query = query.where(File.file_version.in_(args.file_version))
    if args.dobs_start:
        query = query.where(File.date_obs > parse_datetime_str(args.dobs_start))
    if args.dobs_end:
        query = query.where(File.date_obs < parse_datetime_str(args.dobs_end))
    if args.dcreate_start:
        query = query.where(File.date_created > parse_datetime_str(args.dcreate_start))
    if args.dcreate_end:
        query = query.where(File.date_created < parse_datetime_str(args.dcreate_end))

    if args.max_n_files:
        query = query.limit(args.max_n_files)

    files = query.all()

    print(f"Found {len(files)} files")

    if not args.force:
        input("Press enter if that seems right.")

    if any(f.state in ['planning', 'planned', 'creating', 'revivable'] for f in files):
        print("Selected files haves states indicating the pipeline is still running for this flow type.")
        print("Please clear out any planned or running flows and try again.")
        sys.exit()

    errors = []
    with ProcessPoolExecutor(args.workers) as p, tqdm(total=len(files)) as pbar:
        try:
            for i, (file, success_or_msg) in enumerate(zip(files, p.map(productionify_file, files, repeat(config),
                                                           repeat(args.data_root), repeat(old_version_pattern),
                                                           repeat(new_version), chunksize=2))):
                pbar.update()
                if success_or_msg is True:
                    file.file_version = new_version
                else:
                    errors.append(success_or_msg)
                if i % 200 == 0:
                    session.commit()
        except KeyboardInterrupt:
            pass
        session.commit()
        print("Repeats of all error messages:")
        print('\n'.join(errors))
