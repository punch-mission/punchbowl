import re
import sys
import argparse
import pandas as pd

from sqlalchemy import text
from datetime import datetime
from dateutil.parser import parse
from collections import defaultdict
from punchbowl.auto.control.util import get_database_session  

def get_replay_requests(packet_group, reasons_for_replay):

    replay_requests = []
    replay_flashblock_start = None
    previous_reason = None

    for packet in packet_group:

        current_reason = packet["last_skip_reason"]

        is_failure = any(
            re.search(reason, str(current_reason))
            for reason in reasons_for_replay
        )

        if is_failure:

            if current_reason != previous_reason:

                # Close previous failure sequence if one exists
                if replay_flashblock_start is not None:
                    replay_flashblock_end = previous_flashblock

                    flash_blocks_needed = max(
                        1,
                        int(replay_flashblock_end) - int(replay_flashblock_start)
                    )

                    replay_requests.append(
                        f"start mops_fsw_start_fast_replay"
                        f"(xfi,{replay_flashblock_start},{flash_blocks_needed})"
                    )

                # Start a new failure sequence
                replay_flashblock_start = packet["flash_block"]

        else:

            # Close current failure sequence
            if replay_flashblock_start is not None:

                replay_flashblock_end = previous_flashblock

                flash_blocks_needed = max(
                    1,
                    int(replay_flashblock_end) - int(replay_flashblock_start)
                )

                replay_requests.append(
                    f"start mops_fsw_start_fast_replay"
                    f"(xfi,{replay_flashblock_start},{flash_blocks_needed})"
                )

                replay_flashblock_start = None

        previous_reason = current_reason
        previous_flashblock = packet["flash_block"]

    # Handle a failure sequence that extends to the final packet
    if replay_flashblock_start is not None:

        replay_flashblock_end = packet_group[-1]["flash_block"]

        flash_blocks_needed = max(
            1,
            int(replay_flashblock_end) - int(replay_flashblock_start)
        )

        replay_requests.append(
            f"start mops_fsw_start_fast_replay"
            f"(xfi,{replay_flashblock_start},{flash_blocks_needed})"
        )

    return replay_requests

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
 
    parser.add_argument(
        '-s', '--start',
        type=parse,
        required=True,
        help='Input start date in "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS" format.'
    )

    parser.add_argument(
        '-e', '--end',
        type=parse,
        default=datetime.today(),
        help='Input end date in "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS" format. Defaults to today\'s date'
    )

    args = parser.parse_args()

    # TODO: inputs should be changed to grab any new data that appears with each L0 deploy, rather than using user arguments 
    start_date = args.start
    end_date = args.end

    # Create database session
    session, engine = get_database_session(get_engine=True)

    # Craft query
    query = text("""
        SELECT flash_block, spacecraft_id, ccsds_sequence_count, timestamp, last_skip_reason
        FROM sci_xfi
        WHERE timestamp >= :start_date
            AND timestamp <= :end_date
        ORDER BY spacecraft_id, timestamp;
    """)

    # Execute query
    result = session.execute(
        query,
        {
            "start_date": start_date,
            "end_date": end_date
        }
    )

    rows = result.fetchall()

    # Create packet dictionary
    packets = defaultdict(list)

    for row in rows:

        scid = row.spacecraft_id

        packets[scid].append({
            "flash_block": row.flash_block,
            "spacecraft_id": row.spacecraft_id,
            "ccsds_sequence_count": row.ccsds_sequence_count,
            "timestamp": row.timestamp,
            "last_skip_reason": row.last_skip_reason
        })

    # List reasons to request a replay
    reasons_for_replay = [
        r"Waiting for replay",
        r"Image decoding failed charls_jpegls_decoder*"
    ]

    scids = {
        int("0x10",16): "WFI01",
        int("0x2F",16): "NFI00",
        int("0x2C",16): "WFI02",
        int("0xF9",16): "WFI03"
    }

    # Generate replay request files
    today = datetime.today().strftime("%Y_%m_%d")

    for scid, packet_group in packets.items():

        filename = f"{scids[scid]}_{today}_ReplayRequest.txt"

        replay_requests = get_replay_requests(packet_group, reasons_for_replay)

        with open(filename, "w") as file:
            file.write("\n".join(replay_requests))

    #TODO: drop these files in the designated location on Griffin for MOC pickup 