import os
import sqlite3
import argparse
from datetime import datetime

import pandas as pd
from prefect_sqlalchemy import SqlAlchemyConnector
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from punchbowl.auto.control.util import get_database_session

if __name__ == "__main__":

    ## Argparse setup ##
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--start', type=lambda s: datetime.strptime(s, "%Y-%m-%d"), required=True, help='Input start date in YYYY-MM-DD format')
    parser.add_argument('-e','--end', type=lambda s: datetime.strptime(s, "%Y-%m-%d"), default=datetime.today(), help="Input end date in YYYY-MM-DD format. Defaults to today's date")
    args = parser.parse_args()

    start_date = args.start
    end_date = args.end

    ## Database session setup ##
    # host this on the server in the future and use this setup rather than the local setup to access db?
    #session, engine = get_database_session(get_engine=True)

    # local db access setup
    engine = create_engine(
        f"mysql+pymysql://soc_operator:gopunch!@localhost:3307/punchpipe",
        pool_pre_ping=True,
    )
    SessionLocal = sessionmaker(bind=engine)

    ## Query Crafting ##
    bad_packets_query = text("""
        SELECT flash_block, spacecraft_id, ccsds_sequence_count, timestamp, last_skip_reason
        FROM sci_xfi
        WHERE timestamp >= :start_date
            AND timestamp <= :end_date
            AND last_skip_reason LIKE 'Waiting for replay%'
        """)

    ## Query the database ##
    with SessionLocal() as session: # local db session
        df = (
            pd.read_sql(
                bad_packets_query,
                session.connection(),
                params = {
                    "start_date": start_date,
                    "end_date": end_date
                    }
                )
        )

    df.to_csv('test.csv', index=False)
