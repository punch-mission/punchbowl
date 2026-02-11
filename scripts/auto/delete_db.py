from prefect_sqlalchemy import SqlAlchemyConnector

from punchbowl.auto.control.db import Base

if __name__ == "__main__":
    credentials = SqlAlchemyConnector.load("mariadb-creds")
    engine = credentials.get_engine()
    Base.metadata.drop_all(bind=engine)
