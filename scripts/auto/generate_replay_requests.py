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
  
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--start', type=lambda s: datetime.strptime(s, "%Y-%m-%d"), required=True, 
                        help='Input start date in YYYY-MM-DD format')
    parser.add_argument('-e','--end', type=lambda s: datetime.strptime(s, "%Y-%m-%d"), default=datetime.today(), 
                        help="Input end date in YYYY-MM-DD format. Defaults to today's date")
    args = parser.parse_args()

    start_date = args.start
    end_date = args.end

    # db session setup 
    session, engine = get_database_session(get_engine=True)

    print("Database connected")

    print("Executing query...")

    # Query crafting 
    bad_packets_query = text("""
        SELECT flash_block, spacecraft_id, ccsds_sequence_count, timestamp, last_skip_reason
        FROM sci_xfi
        WHERE timestamp >= :start_date
            AND timestamp <= :end_date
            AND (last_skip_reason LIKE 'Waiting for replay%'
            OR last_skip_reason LIKE 'Image decoding failed charls_jpegls_decoder_%')
        """)

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

    print(f"Query finished. Retrieved {len(df)} records.")

    df.to_csv('test.csv', index=False)
