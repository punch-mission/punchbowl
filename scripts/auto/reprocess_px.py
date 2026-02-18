"""Triggers all PX files to be remade."""

from datetime import timedelta

from prefect_sqlalchemy import SqlAlchemyConnector
from sqlalchemy import text
from tqdm import tqdm

from punchbowl.auto.control.db import Base

credentials = SqlAlchemyConnector.load("mariadb-creds")
engine = credentials.get_engine()

with engine.connect() as connection:
    get_px_ids = 'select file_id, level, file_type, observatory, date_beg from files where file_type = "PX";'
    px_id_info = connection.execute(text(get_px_ids)).all()
    print(f"Found {len(px_id_info)} PX files")

    get_other_ids = 'select file_id, level, file_type, observatory, date_beg from files where file_type != "PX" and level = "0";'
    not_px_id_info = connection.execute(text(get_other_ids)).all()

    not_px_set = set([(e[-1], e[3]) for e in not_px_id_info]) # date_beg and observatory for each entry
    px_id_info = [e for e in px_id_info if (e[-1], e[3]) not in not_px_set]
    print(f"Found {len(px_id_info)} PX files without matching real files")

    observatory2spacecraft = {'1':16, '2': 44, '3': 249, '4': 47}
    all_packet_ids = []
    for file_id, _, _, observatory, date_beg in tqdm(px_id_info):
        packet_reference_time = date_beg - timedelta(seconds=3.8)  # this accounts for the time to clear

        start_time = packet_reference_time - timedelta(seconds=1)
        start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")

        end_time = packet_reference_time + timedelta(seconds=1)
        end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")

        spacecraft = observatory2spacecraft[observatory]
        this_query = (f'select id, tlm_id, spacecraft_id, packet_index, timestamp, is_used, packet_group '
                      f'from sci_xfi where timestamp between "{start_str}" and "{end_str}" and spacecraft_id = {spacecraft}')
        packets_to_update_info = connection.execute(text(this_query)).all()
        packet_ids = [p[0] for p in packets_to_update_info]
        all_packet_ids.extend(packet_ids)
    print(f"Will reset {len(all_packet_ids)} packets")
    update_query = f'update sci_xfi set is_used=0 where id in {tuple(all_packet_ids)}'
    connection.execute(text(update_query))
    connection.execute(text("commit"))
