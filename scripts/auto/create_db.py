from prefect_sqlalchemy import SqlAlchemyConnector
from sqlalchemy import text

from punchbowl.auto.control.db import Base

if __name__ == "__main__":

    credentials = SqlAlchemyConnector.load("mariadb-creds")
    engine = credentials.get_engine()
    with engine.connect() as connection:
        result = connection.execute(text('CREATE DATABASE IF NOT EXISTS punchpipe;'))
    Base.metadata.create_all(engine)
