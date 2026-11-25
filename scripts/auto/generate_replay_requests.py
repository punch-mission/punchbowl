import re
import argparse
from datetime import datetime
from collections import defaultdict

from dateutil.parser import parse
from sqlalchemy import text

from punchbowl.auto.control.util import get_database_session
from punchbowl.constants import SPACECRAFT_ID_HEX


def generate_replay_requests(start_date: datetime, end_date: datetime):
    """
    Query the punchpipe database for L0 packet failures and generate
    spacecraft-specific replay request text files.

    Parameters
    ----------
        start_date : datetime
            Beginning of the database query time range.
        end_date : datetime
            End of the database query time range.

    Returns
    -------
        None: Writes timestamped replay request text files to the
            current working directory.

    """

    # List reasons to request a replay (regex)
    reasons_for_replay = [
        r"Waiting for replay",
        r"^Image decoding failed charls_jpegls_decoder.*"
    ]

    # Create database session
    session, engine = get_database_session(get_engine=True)

    # Craft query
    from punchbowl.auto.control.db import SCI_XFI
    rows = (session.query(SCI_XFI)
                   .where(SCI_XFI.timestamp >= start_date)
                   .where(SCI_XFI.timestamp <= end_date)
                   .order_by(SCI_XFI.spacecraft_id, SCI_XFI.timestamp)
                   .all())
    session.close()

    if not rows:
        print('No packets found for the specified date range')
        return

    # Create packet dictionary grouped by spacecraft ID
    packets = defaultdict(list)

    for row in rows:

        spacecraft_id = row.spacecraft_id

        packets[spacecraft_id].append({
            "flash_block": row.flash_block,
            "spacecraft_id": row.spacecraft_id,
            "ccsds_sequence_count": row.ccsds_sequence_count,
            "timestamp": row.timestamp,
            "last_skip_reason": row.last_skip_reason
        })

    # Generate timestamp for replay request filenames
    timestamp_now = datetime.today().strftime("%j_%H_%M")

    # Generate replay request files for each spacecraft
    for spacecraft_id, packet_group in packets.items():

        replay_requests = []

        replay_flashblock_start = None
        previous_reason = None

        # Iterate through packets for this spacecraft
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
                            int(replay_flashblock_end)
                            - int(replay_flashblock_start)
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

                    flash_blocks_needed = int(replay_flashblock_end) + 1 - int(replay_flashblock_start)


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

            flash_blocks_needed = int(replay_flashblock_end) + 1 - int(replay_flashblock_start)

            replay_requests.append(
                f"start mops_fsw_start_fast_replay"
                f"(xfi,{replay_flashblock_start},{flash_blocks_needed})"
            )

        # Create spacecraft-specific output file
        if SPACECRAFT_ID_HEX.get(spacecraft_id) is not None and len(replay_requests) > 0:

            filename = f"{SPACECRAFT_ID_HEX.get(spacecraft_id)}_{timestamp_now}_ReplayRequest.txt"

            with open(filename, "w") as file:
                file.write("\n".join(replay_requests))

            print(f"Generated {filename} with {len(replay_requests)} replay requests.")

    # TODO: Drop these files in the designated location on Griffin for MOC pickup


if __name__ == "__main__":

    # TODO: Incorporate this into L0 flow rather than using it as a standalone script

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-s', '--start',
        type=parse,
        required=True,
        help='Input start date in YYYY-MM-DD or YYYY-MM-DD HH:MM:SS format.'
    )

    parser.add_argument(
        '-e', '--end',
        type=parse,
        default=datetime.today(),
        help='Input end date in YYYY-MM-DD or YYYY-MM-DD HH:MM:SS format. Defaults to today.'
    )

    args = parser.parse_args()

    generate_replay_requests(args.start, args.end)
