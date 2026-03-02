import os
import pickle
import argparse
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass

from dateutil.parser import parse as parse_datetime_str
from sqlalchemy.orm import aliased
from tqdm import tqdm

from punchbowl.auto.control.db import File, FileRelationship, Flow
from punchbowl.auto.control.util import get_database_session


@dataclass
class ExportedFlow:
    flow_level: str
    flow_type: str
    flow_run_name: str
    flow_run_id: str
    state: str
    creation_time: datetime
    launch_time: datetime
    start_time: datetime
    end_time: datetime
    priority: int

    @staticmethod
    def from_Flow(flow: Flow):
        return ExportedFlow(
            flow_level=flow.flow_level,
            flow_type=flow.flow_type,
            flow_run_name=flow.flow_run_name,
            flow_run_id=flow.flow_run_id,
            state=flow.state,
            creation_time=flow.creation_time,
            launch_time=flow.launch_time,
            start_time=flow.start_time,
            end_time=flow.end_time,
            priority=flow.priority,
        )


@dataclass
class ExportedFile:
    level: str
    file_type: str
    observatory: str
    file_version: str
    software_version: str
    date_created: datetime
    date_obs: datetime
    date_beg: datetime
    date_end: datetime
    polarization: str
    state: str
    crota: float
    outlier: int
    bad_packets: bool
    processing_flow: ExportedFlow
    parents: list[tuple]
    _directory: str
    _filename: str

    def find_in_db(self, session):
        query = session.query(File)
        for col in ['level', 'file_type', 'observatory', 'file_version', 'date_obs', 'polarization']:
            if getattr(self, col) is not None:
                query = query.where(getattr(File, col) == getattr(self, col))
        try:
            return query.one_or_none()
        except:
            print(self.level, self.file_type, self.observatory, self.date_obs, self.state)
            raise

    @staticmethod
    def export_File(file: File, session):
        if file.processing_flow is None:
            processing_flow = None
        else:
            processing_flow = session.query(Flow).where(Flow.flow_id == file.processing_flow).one()

        child = aliased(File)
        parent = aliased(File)
        parents = (session
                   .query(parent)
                   .join(FileRelationship, FileRelationship.parent == parent.file_id)
                   .join(child, FileRelationship.child == child.file_id)
                   .filter(child.file_id == file.file_id).all())

        parents = [(File_to_find_key(p), p.state) for p in parents]

        return ExportedFile(
            level=file.level,
            file_type=file.file_type,
            observatory=file.observatory,
            file_version=file.file_version,
            software_version=file.software_version,
            date_created=file.date_created,
            date_obs=file.date_obs,
            date_beg=file.date_beg,
            date_end=file.date_end,
            polarization=file.polarization,
            state=file.state,
            crota=file.crota,
            outlier=file.outlier,
            bad_packets=file.bad_packets,
            processing_flow=ExportedFlow.from_Flow(processing_flow) if processing_flow is not None else None,
            parents=parents,
            _directory=file.directory(''),
            _filename=file.filename(),
        )

    def filename(self):
        return self._filename

    def directory(self, root_dir):
        return os.path.join(root_dir, self._directory)

    def add_to_db(self, session):
        added_files = 0
        added_flows = 0
        added_relationships = 0

        new_file = self.find_in_db(session)
        if new_file is None:
            new_file = File(
                level=self.level,
                file_type=self.file_type,
                observatory=self.observatory,
                file_version=self.file_version,
                software_version=self.software_version,
                date_created=self.date_created,
                date_obs=self.date_obs,
                date_beg=self.date_beg,
                date_end=self.date_end,
                polarization=self.polarization,
                state=self.state,
                crota=self.crota,
                outlier=self.outlier,
                bad_packets=self.bad_packets,
            )
            session.add(new_file)
            added_files = 1

            if self.processing_flow is not None:
                new_flow = Flow(
                    flow_level=self.processing_flow.flow_level,
                    flow_type=self.processing_flow.flow_type,
                    flow_run_name=self.processing_flow.flow_run_name,
                    flow_run_id=self.processing_flow.flow_run_id,
                    state=self.processing_flow.state,
                    creation_time=self.processing_flow.creation_time,
                    launch_time=self.processing_flow.launch_time,
                    start_time=self.processing_flow.start_time,
                    end_time=self.processing_flow.end_time,
                    priority=self.processing_flow.priority,
                )
                session.add(new_flow)
                added_flows = 1

            session.commit()

            if self.processing_flow is not None:
                new_file.processing_flow = new_flow.flow_id

        for parent_key, parent_state in self.parents:
            parent = find_key_to_File(parent_key, session)

            if parent is not None:
                # Stray light, F-corona, and starfield models shouldn't affect the parents' state
                if self.file_type[0] != 'S' and self.file_type not in ['PF', 'CF', 'PS', 'CS']:
                    parent.state = parent_state
                if not (session.query(FileRelationship)
                        .where(FileRelationship.parent == parent.file_id)
                        .where(FileRelationship.child == new_file.file_id).first()):
                    relationship = FileRelationship(parent=parent.file_id, child=new_file.file_id)
                    session.add(relationship)
                    added_relationships += 1

        session.commit()

        return added_files, added_flows, added_relationships


def File_to_find_key(file: File):
    return (file.level, file.file_type, file.observatory, file.file_version, file.date_obs, file.polarization)


def find_key_to_File(key: tuple, session):
    level, file_type, observatory, file_version, date_obs, polarization = key
    try:
        return (session.query(File)
                .where(File.level == level)
                .where(File.file_type == file_type)
                .where(File.observatory == observatory)
                .where(File.file_version == file_version)
                .where(File.date_obs == date_obs)
                .where(File.polarization == polarization)
                ).one_or_none()
    except:
        print(key)
        raise


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
    parser.add_argument('--data', action='store_true')
    parser.add_argument('--db', action='store_true')
    parser.add_argument('mode')
    parser.add_argument('dump_path')
    parser.add_argument('data_root', default=None, nargs='?')

    args = parser.parse_args()

    if args.mode == 'export':
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

        files = [ExportedFile.export_File(f, session) for f in tqdm(files)]

        print(f"Got {len(files)} files")

        export = files

        if args.data:
            print("Collecting data...")
            export = []
            for file in tqdm(files):
                path = os.path.join(file.directory(args.data_root), file.filename())
                with open(path, 'rb') as f:
                    data = f.read()
                export.append((data, file))

        with open(args.dump_path, 'wb') as f:
            pickle.dump(export, f)

    elif args.mode == 'import':
        with open(args.dump_path, 'rb') as f:
            files = pickle.load(f)

        if args.db:
            session = get_database_session()

        print(f"Importing {len(files)} files")

        n_added_files, n_added_flows, n_added_relationships, n_written = 0, 0, 0, 0

        for file in tqdm(files):
            if isinstance(file, tuple):
                if args.db:
                    session.commit()
                data, file = file
                path = os.path.join(file.directory(args.data_root), file.filename())
                if os.path.exists(path):
                    print(f"Skipping already-existing {path}")
                    continue
                with open(path, 'wb') as f:
                    f.write(data)
                    n_written += 1
            if args.db:
                added_file, added_flow, added_relationships = file.add_to_db(session)
                n_added_files += added_file
                n_added_flows += added_flow
                n_added_relationships += added_relationships
        print(f"Added {n_added_files} files, {n_added_flows} flows, {n_added_relationships} relationships, wrote {n_written} files")
    elif args.mode == 'info':
        with open(args.dump_path, 'rb') as f:
            files = pickle.load(f)
        levels = defaultdict(lambda: 0)
        types = defaultdict(lambda: 0)
        obs = defaultdict(lambda: 0)
        versions = defaultdict(lambda: 0)

        for file in files:
            if isinstance(file, tuple):
                _, file = file
            levels[file.level] += 1
            types[file.file_type] += 1
            obs[file.observatory] += 1
            versions[file.file_version] += 1

        print("--- Level ---")
        for level in sorted(levels.keys()):
            print(f"{levels[level]:7d} L{level}")

        print("--- Type ---")
        for level in sorted(types.keys()):
            print(f"{types[level]:7d} {level}")

        print("--- Observatory ---")
        for o in sorted(obs.keys()):
            print(f"{obs[o]:7d} {o}")

        print("--- Version ---")
        for v in sorted(versions.keys()):
            print(f"{versions[v]:7d} {v}")
    else:
        print("Unrecognized mode")
