import os
import argparse
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

if __name__ == "__main__":

    #might want to add a start date validation function
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--start', type=lambda s: datetime.strptime(s, "%Y-%m-%d"), required=True, help='Input start date in YYYY-MM-DD format')
    parser.add_argument('-e','--end', type=lambda s: datetime.strptime(s, "%Y-%m-%d"), default=datetime.today(), help="Input end date in YYYY-MM-DD format. Defaults to today's date")
    args = parser.parse_args()

    #we can set this db access up differently depending on if we want this to run automatically or be run by the daily operator
    engine = create_engine(
        f"mysql+pymysql://soc_operator:gopunch!@10.100.12.25:3306/punchpipe",
        pool_pre_ping=True,
    )

    Session = sessionmaker(bind=engine)
