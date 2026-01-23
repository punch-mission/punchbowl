import os
from math import inf
from datetime import UTC, datetime
from itertools import islice

import yaml
from ndcube import NDCube
from prefect.variables import Variable
from prefect_sqlalchemy import SqlAlchemyConnector
from sqlalchemy import or_
from sqlalchemy.orm import Session
from yaml.loader import FullLoader

from punchbowl.data import get_base_file_name, write_ndcube_to_fits, write_ndcube_to_quicklook
from punchbowl.pipe.control.db import File

DEFAULT_SCALING = (5e-13, 5e-11)

def get_database_session(get_engine=False, engine_kwargs={}):
    """Sets up a session to connect to the MariaDB punchpipe database"""
    credentials = SqlAlchemyConnector.load("mariadb-creds", _sync=True)
    engine = credentials.get_engine(**engine_kwargs)
    session = Session(engine)

    if get_engine:
        return session, engine
    return session


def update_file_state(session, file_id, new_state):
    session.query(File).where(File.file_id == file_id).update({"state": new_state})
    session.commit()


def load_pipeline_configuration(path: str = None) -> dict:
    if path is None:
        path = Variable.get("punchpipe_config", "punchpipe_config.yaml")
    with open(path) as f:
        config = yaml.load(f, Loader=FullLoader)
    # TODO: add validation
    return config


def load_quicklook_scaling(level: str = None, product: str = None, obscode: str = None, path: str = None) -> (float, float):
    if path is None:
        path = Variable.get("punchpipe_config", "punchpipe_config.yaml")
    with open(path) as f:
        config = yaml.load(f, Loader=FullLoader)
    if "quicklook_scaling" in config:
        if level:
            level_data = config.get("quicklook_scaling", {}).get(level, {})
            if product and isinstance(level_data, dict):
                product_data = level_data.get(product, level_data.get("default"))
                if obscode == "4":
                    return product_data[1]
                return product_data[0]
            if obscode == "4":
                return level_data.get("default")[1]
            return level_data.get("default")[0]
        return DEFAULT_SCALING
    return DEFAULT_SCALING


def write_file(data: NDCube, corresponding_file_db_entry, pipeline_config) -> None:
    output_filename = os.path.join(
        corresponding_file_db_entry.directory(pipeline_config["root"]), corresponding_file_db_entry.filename(),
    )
    output_dir = os.path.dirname(output_filename)
    os.makedirs(output_dir, exist_ok=True)
    write_ndcube_to_fits(data, output_filename)

    # TODO - Configure to write each layer separately?
    layer = 0 if len(data.data.shape) > 2 else None
    ql_directory = pipeline_config.get("ql_root", pipeline_config["root"])
    ql_filename = os.path.join(corresponding_file_db_entry.directory(ql_directory),
                                                                     corresponding_file_db_entry.filename())
    ql_filename = ql_filename.replace(".fits", ".jp2")
    write_ndcube_to_quicklook(data, ql_filename, layer=layer)
    return output_filename


def match_data_with_file_db_entry(data: NDCube, file_db_entry_list):
    # figure out which file_db_entry this corresponds to
    matching_entries = [
        file_db_entry
        for file_db_entry in file_db_entry_list
        if file_db_entry.filename() == get_base_file_name(data) + ".fits"
    ]
    if len(matching_entries) == 0:
        for file in file_db_entry_list:
            raise RuntimeError(f"There did not exist a file_db_entry for this output cube: "
                               f"result={get_base_file_name(data)}. Candidate: {file.filename()}")
    elif len(matching_entries) > 1:
        raise RuntimeError("There were many database entries matching this result. There should only be one.")
    else:
        return matching_entries[0]


def get_files_in_time_window(level: str,
                             file_type: str,
                             obs_code: str,
                             start_time: datetime,
                             end_time: datetime,
                             session: Session | None) -> list[File]:
    if session is None:
        get_database_session()

    return (session.query(File).filter(or_(File.state == "created", File.state == "progressed"))
            .filter(File.level == level)
            .filter(File.file_type == file_type)
            .filter(File.observatory == obs_code)
            .filter(File.date_obs > start_time)
            .filter(File.date_obs <= end_time).all())


def batched(iterable, n):
    # batched('ABCDEFG', 3) → ABC DEF G
    # This is basically itertools.batched, but that only exists in Python >= 3.12
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


def group_files_by_time(files: list[File],
                        max_duration_seconds: float = inf,
                        max_per_group: int = inf) -> list[list[File]]:
    # We need to group up files by date_obs, but we need to handle small variations in date_obs. The files are coming
    # from the database already sorted, so let's just walk through the list of files and cut a group boundary every time
    # date_obs increases by more than a threshold.
    grouped_files = []
    # We'll keep track of where the current group started, and then keep stepping to find the end of this group.
    group_start = 0
    tstamp_start = files[0].date_obs.replace(tzinfo=UTC).timestamp()
    file_under_consideration = 0
    while True:
        file_under_consideration += 1
        if file_under_consideration == len(files):
            break
        this_tstamp = files[file_under_consideration].date_obs.replace(tzinfo=UTC).timestamp()
        if (abs(this_tstamp - tstamp_start) > max_duration_seconds
                or file_under_consideration - group_start >= max_per_group):
            # date_obs has jumped by more than our tolerance, so let's cut the group and then start tracking the next
            # one
            grouped_files.append(files[group_start:file_under_consideration])
            group_start = file_under_consideration
            tstamp_start = this_tstamp
    grouped_files.append(files[group_start:])
    return grouped_files
