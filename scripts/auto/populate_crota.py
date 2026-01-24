import os

import sqlalchemy
from astropy.io import fits
from sqlalchemy import Column, Float, text
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from punchbowl.auto.control.db import File
from punchbowl.auto.control.util import get_database_session


def add_column(session, table_name, column):
    column_name = column.compile(dialect=engine.dialect)
    column_type = column.type.compile(engine.dialect)
    session.execute(text('ALTER TABLE %s ADD COLUMN %s %s' % (table_name, column_name, column_type)))

def get_crota_from_file(file):
    try:
        path = os.path.join(file.directory("/d0/punchsoc/real_data/"), file.filename())  # TODO this is only for 190
        header = fits.getheader(path, 1)
        return file, header['CROTA']
    except:
        return file, None

if __name__ == "__main__":
    session, engine = get_database_session(get_engine=True)

    #column = Column('crota', Float, nullable=True)
    #add_column(session, "files", column)

    existing_files = (session.query(File)
                      .filter(File.level=="0")
                      .filter(File.file_type.in_(["CR", "PM", "PZ", "PP"]))
                      .filter(File.crota.is_(None)).all())

    for file, crota in process_map(get_crota_from_file, existing_files, chunksize=100):
        file.crota = crota

    session.commit()
