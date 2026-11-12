import re
import sys
import argparse
import pandas as pd

from sqlalchemy import text
from datetime import datetime
from dateutil.parser import parse
from collections import defaultdict
from punchbowl.auto.control.util import get_database_session  

if __name__ == "__main__":
  
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--start', type=parse, required=True, 
                        help='Input start date in "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS" format.')
    parser.add_argument('-e','--end', type=parse, default=datetime.today(), 
                        help='Input end date in "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS" format. Defaults to today\'s date')
    args = parser.parse_args()

    start_date = args.start
    end_date = args.end

    # Create db session 
    session, engine = get_database_session(get_engine=True)

    print("Database connected")
    print("Executing query...")

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

    print(f"Query finished. Retrieved {len(rows)} packets.")

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

    # Core logic 
    for scid, group in packets.items(): # compare packets with the same spacecraft id

        previous_reason = None 
        failure_count = 0

        for index, packet in enumerate(group): # iterate through packets

            current_reason = packet["last_skip_reason"]
            is_failure = any(re.search(reason, str(packet["last_skip_reason"])) for reason in reasons_for_replay) # if the last skip reason is worthy of replay

            if is_failure and current_reason == previous_reason: # if there are consecutive failures with the same reason

                failure_count += 1 

            elif is_failure: # if there is a failure and this is the first instance 

                #print(f"Failure started at {packet["flash_block"]}") # testing
                replay_flashblock_start = packet["flash_block"] # this flashblock is should be the start of a replay request
                failure_count = 1

            else: 

                if failure_count > 0: # if you break out of the current instance of consecutive failures 

                    replay_flashblock_end = group[index - 1]["flash_block"] # this flashblock should be the end of a replay request 
                    flash_blocks_needed = int(replay_flashblock_end) - int(replay_flashblock_start)
                    #print(f"Failure ended at {group[index -1]["flash_block"]}") # testing 
                    replay_request = f"{scid}: start mops_fsw_start_fast_replay(xfi,{replay_flashblock_start},{flash_blocks_needed})"
                    print(replay_request) # testing

                failure_count = 0 

            previous_reason = current_reason