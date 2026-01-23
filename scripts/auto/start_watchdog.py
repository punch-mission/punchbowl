import itertools
from datetime import datetime, timedelta

from prefect import flow, get_run_logger
from prefect.schedules import Cron

from punchbowl.auto.control.db import File
from punchbowl.auto.control.util import get_database_session, load_pipeline_configuration

LEVEL_0_SPACECRAFT = ["1", "2", "3", "4"]
LEVEL_0_PREFIXES = ["CR", "PM", "PP", "PZ"]
LEVEL_0_CODES = [f"L0_{prefix}{spacecraft}" for prefix, spacecraft in
                 itertools.product(LEVEL_0_PREFIXES, LEVEL_0_SPACECRAFT)]

LEVEL_1_SPACECRAFT = ["1", "2", "3", "4"]
LEVEL_1_PREFIXES = ["CR", "PM", "PP", "PZ",
                    "XM", "XR", "XP", "XZ",
                    "YM", "YR", "YP", "YZ",
                    "TR", "TZ", "TP", "TM",
                    "SR", "SZ", "SP", "SM"]
LEVEL_1_CODES = [f"L1_{prefix}{spacecraft}" for prefix, spacecraft in
                 itertools.product(LEVEL_1_PREFIXES, LEVEL_1_SPACECRAFT)]


LEVEL_2_SPACECRAFT = ["M"]
LEVEL_2_PREFIXES = ["PT", "CT"]
LEVEL_2_CODES = [f"L2_{prefix}{spacecraft}" for prefix, spacecraft in
                 itertools.product(LEVEL_2_PREFIXES, LEVEL_2_SPACECRAFT)]

LEVEL_3_SPACECRAFT = ["M"]
LEVEL_3_PREFIXES = ["PI", "CI", "PF", "CF"]
LEVEL_3_CODES = [f"L3_{prefix}{spacecraft}" for prefix, spacecraft in
                 itertools.product(LEVEL_3_PREFIXES, LEVEL_3_SPACECRAFT)]

LEVEL_Q_CODES = ["LQ_CNN", "LQ_CFN", "LQ_CFM", "LQ_CTM"]

def get_file_count_in_db(level, product, start_time, end_time, session=None):
    if session is None:
        session = get_database_session()

    resulting_files = (
        session.query(File)
         .filter(File.level == level)
         .filter(File.file_type == product[:2])
         .filter(File.observatory == product[2])
         .filter(File.date_created > start_time)
         .filter(File.date_created < end_time)
    ).all()
    return len(resulting_files)

def get_minimum_count(configuration, product_code):
    return configuration.get("minimum_counts", {}).get(product_code, 1)

def get_product_counts(product_codes, start_time, end_time, session=None):
    return {product_code: get_file_count_in_db(product_code[1], product_code[-3:], start_time, end_time, session=session)
     for product_code in product_codes}

@flow
def check_product_counts_watchdog(start_time=None, end_time=None):
    if start_time is None:
        start_time = datetime.now()-timedelta(days=1)

    if end_time is None:
        end_time = datetime.now()

    logger = get_run_logger()

    logger.info(f"Checking product counts for {start_time} -> {end_time}")

    session = get_database_session()

    error_product_codes = []
    configuration = load_pipeline_configuration()
    for level_codes in [LEVEL_0_CODES, LEVEL_1_CODES, LEVEL_2_CODES, LEVEL_3_CODES, LEVEL_Q_CODES]:
        counts = get_product_counts(level_codes, start_time, end_time, session=session)
        for product_code, count in counts.items():
            minimum_expected_count = get_minimum_count(configuration, product_code)
            if count < minimum_expected_count:
                logger.error(f"{product_code} has {count} files instead >= {minimum_expected_count}!")
                error_product_codes.append(product_code)
            else:
                logger.info(f"{product_code} has {count} files.")

    if error_product_codes:
        raise RuntimeError(f"Files were missing for the following codes: {error_product_codes}. "
                           f"See logs for more exhaustive information.")

if __name__ == "__main__":
    check_product_counts_watchdog.serve("watchdog", schedule=Cron("0 7 * * *", timezone="America/Denver"))
