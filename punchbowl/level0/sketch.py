"""
This file contains the sketches of how this code will work in the long run. It is not meant to be complete yet.
"""
import numpy as np
import pandas as pd
from ccsdspy import PacketField
from ccsdspy.decode import _decode_fixed_length

# load the excel overview that describes all the APIDs
overview = pd.read_excel(
    "CYGNSS_TLM.xls",
    sheet_name="Overview",
    names=["name", "apid", "size", "description", "apid_decimal"],
)

# Figure out the packet length of each APID for quick reference later
packet_length = {}
for row in overview.iterrows():
    packet_length[row[1]["apid_decimal"]] = row[1]["size"]

packet_length[
    1152
] = 140  # for CYGNSS data this is a manual necessary change to get the code to run

# Read in the telemetry file
with open("CYGNSS_F7_L0_2022_086_10_15_V01_F.tlm", "rb") as f:
    data = f.read()

# Read in all the packet data! This is the last step before actually decoding
i = 0
apid_mask = 0b00011111111111
file_packet_apids = []
file_length = len(data)
apid_data = {apid: [] for apid in packet_length}
while True:
    this_apid = int(data[i : i + 2].hex(), base=16) & apid_mask
    length = int(data[i + 4 : i + 6].hex(), base=16)
    file_packet_apids.append((i, this_apid, length))
    apid_data[this_apid].append(
        np.array(
            [int(e, base=16) for e in data[i : i + length + 7].hex(" ").split()]
        ).astype("uint8")
    )
    i += length + 7  # packet_length[this_apid]
    if i >= file_length:
        break

# Load the specific definitions for the desired APID
eng_adcsio = pd.read_excel("CYGNSS_TLM.xls", sheet_name="ENG_ADCSIO")
diag_nst_raw = pd.read_excel("CYGNSS_TLM.xls", sheet_name="DIAG_NST_RAW")

# Prepare a CCSDSpy package definition
packet_definition = []
for row in diag_nst_raw.iterrows():
    name = row[1][0]
    kind = row[1][2]
    kind = "uint" if row[1][2][0] == "U" else "int"
    start_byte = row[1][6]
    start_bit = row[1][7]
    size = row[1][8]
    packet_definition.append(PacketField(name=name, data_type=kind, bit_length=size))

# do the decoding of all packets of the specific APID type
decoded = _decode_fixed_length(
    np.concatenate(apid_data[1152]), packet_definition
).keys()
