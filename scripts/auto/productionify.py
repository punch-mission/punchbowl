import os
import re
import argparse

from astropy.io import fits
from dateutil.parser import parse as parse_datetime_str
from tqdm import tqdm

from punchbowl.auto.control.db import File, FileRelationship, Flow
from punchbowl.auto.control.util import _write_quicklook, get_database_session, load_pipeline_configuration
from punchbowl.data.punch_io import _make_provenance_hdu, load_ndcube_from_fits, write_file_hash


def replace_version(pattern, replacement, string):
    return re.sub(f"_v{pattern}_", f"_v{replacement}", string)


def update_metadata(path, old_pattern, new_version):
    with fits.open(path, mode='update', disable_image_compression=True) as hdul:
        for i, hdu in enumerate(hdul):
            if hdu['EXTNAME'] in ('PRIMARY DATA ARRAY', 'UNCERTAINTY ARRAY'):
                for key in hdu.header:
                    hdu.header[key] = replace_version(old_pattern, new_version, hdu.header[key])
            elif hdu['EXTNAME'] == 'FILE PROVENANCE':
                source_files = hdu.data['provenance']
                source_files = [replace_version(old_pattern, new_version, f) for f in source_files]
                hdu_provenance = _make_provenance_hdu(source_files)
                hdul[i] = hdu_provenance


def productionify_file(file: File, config: dict, data_root: str, old_pattern, new_version):
    old_path = os.path.join(file.directory(data_root), file.filename())
    update_metadata(old_path, old_pattern, new_version)
    file.version = new_version
    path = os.path.join(file.directory(data_root), file.filename())
    os.rename(old_path, path)
    if os.path.exists(old_path + '.sha'):
        os.rename(old_path + '.sha', path + '.sha')
    else:
        write_file_hash(path)
    if os.path.exists(old_path.replace('.fits', '.jp2')):
        os.rename(old_path.replace('.fits', '.jp2'), path.replace('.fits', '.jp2'))
    else:
        cube = load_ndcube_from_fits(path)
        _write_quicklook(config, file, cube)


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
    parser.add_argument('data_root')
    parser.add_argument('pipeline_config')
    parser.add_argument('old_version_pattern')
    parser.add_argument('new_version')

    args = parser.parse_args()

    pattern = args.old_version_pattern
    if pattern.startswith('v'):
        print("Stripping leading 'v' from old version")
        pattern = pattern[1:]

    replacement = args.new_version
    if replacement.startswith('v'):
        print("Stripping leading 'v' from new version")
        replacement = replacement[1:]

    config = load_pipeline_configuration(args.pipeline_config)
    session = get_database_session()

    query = session.query(File).where(File.state.in_(['created', 'progressed']))
    if args.level:
        query = query.where(File.level.in_(args.level))
    if args.type:
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

    files = query.all()

    print(f"Found {len(files)} files")

    for file in tqdm(files):
        productionify_file(file, config, args.data_root, pattern, replacement)

    session.commit()
