import os

import sqlalchemy
from astropy.io import fits
from sqlalchemy import Column, Float, text

from punchbowl.auto.control.db import File
from punchbowl.auto.control.util import get_database_session


def add_column(session, table_name, column):
    column_name = column.compile(dialect=engine.dialect)
    column_type = column.type.compile(engine.dialect)
    session.execute(text('ALTER TABLE %s ADD COLUMN %s %s' % (table_name, column_name, column_type)))

if __name__ == "__main__":
    session, engine = get_database_session(get_engine=True)


    # Source - https://stackoverflow.com/a
    # Posted by AlexP, modified by community. See post 'Timeline' for change history
    # Retrieved 2025-11-10, License - CC BY-SA 4.0

    column = Column('crota', Float, nullable=True)
    add_column(session, "files", column)

    existing_files = session.query(File).filter(File.level=="0").filter(File.crota.is_(None)).all()

    for file in existing_files:
        path = os.path.join(file.directory("/d0/punchsoc/real_data/"), file.filename()) # TODO this is only for 190
        header = fits.getheader(path, 1)
        file.crota = header["CROTA"]

    session.commit()
