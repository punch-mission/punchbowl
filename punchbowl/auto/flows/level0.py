import io
import os
import json
import base64
import hashlib
import traceback
import multiprocessing
from glob import glob
from typing import Any
from datetime import UTC, datetime, timedelta
from collections import defaultdict
from collections.abc import Callable

import astropy.units as u
import ccsdspy
import numpy as np
import pandas as pd
import pylibjpeg
import quaternion
from astropy.coordinates import GCRS, EarthLocation, HeliocentricMeanEcliptic, SkyCoord
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS
from ccsdspy import PacketArray, PacketField, converters
from ccsdspy.utils import split_by_apid
from dateutil.parser import parse as parse_datetime_str
from ndcube import NDCube
from prefect import flow, get_run_logger, task
from prefect.blocks.core import Block
from prefect.blocks.fields import SecretDict
from prefect.cache_policies import NO_CACHE
from prefect.context import get_run_context
from prefect_sqlalchemy import SqlAlchemyConnector
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session
from sunpy.coordinates import (
    HeliocentricEarthEcliptic,
    HeliocentricInertial,
    HeliographicCarrington,
    HeliographicStonyhurst,
    sun,
)

import punchbowl
from punchbowl import __version__
from punchbowl.auto.control.cache_layer import manager
from punchbowl.auto.control.cache_layer.loader_base_class import LoaderABC
from punchbowl.auto.control.db import (
    ENG_CEB,
    ENG_LED,
    ENG_LZ,
    ENG_PFW,
    ENG_XACT,
    PACKETNAME2SQL,
    SCI_XFI,
    File,
    Flow,
    PacketHistory,
    TLMFiles,
)
from punchbowl.auto.control.util import load_pipeline_configuration
from punchbowl.auto.flows.util import file_name_to_full_path
from punchbowl.data import NormalizedMetadata, get_base_file_name, punch_io, write_ndcube_to_fits
from punchbowl.data.wcs import calculate_helio_wcs_from_celestial, calculate_pc_matrix
from punchbowl.exceptions import MissingMetadataError
from punchbowl.limits import LimitSet
from punchbowl.util import load_mask_file

FIXED_PACKETS = ["ENG_XACT", "ENG_LED", "ENG_PFW", "ENG_CEB", "ENG_LZ"]
VARIABLE_PACKETS = ["SCI_XFI"]
PACKET_CADENCE = {}
SC_TIME_EPOCH = Time(2000.0, format="decimalyear", scale="tai")
NFI_PFW_POSITION_MAPPING = ["PM", "DK", "PZ", "PP", "CR"]
WFI_PFW_POSITION_MAPPING = ["PP", "DK", "PZ", "PM", "CR"]

credentials = SqlAlchemyConnector.load("mariadb-creds", _sync=True)
engine = credentials.get_engine()

def initializer():
    """
    Ensure the parent proc's database connections are not touched
    in the new connection pool
    """
    engine.dispose(close=False)

class SpacecraftMapping(Block):
    mapping: SecretDict

class TaiDatetimeConverter(converters.DatetimeConverter):
    """
    Like the parent class, but takes an astropy Time object (which inherently encodes a
    timescale) instead of a datetime for the `since` initialization argument, and uses astropy
    TimeDelta objects for date math (instead of leapsecond-naïve Python timedeltas). Values are
    treated as offsets on a TAI timescale.
    """

    def __init__(self, since: Time, units: "str|tuple[str]"):
        if not isinstance(since, Time):
            raise TypeError("Argument 'since' must be an instance of astropy.time.Time")

        if isinstance(units, str):
            units_tuple = (units,)
        elif isinstance(units, tuple):
            units_tuple = units
        else:
            raise TypeError("Argument 'units' must be either a string or tuple")

        if not (set(units_tuple) <= set(self._VALID_UNITS)):
            raise ValueError("One or more units are invalid")

        self._since = since
        self._units = units_tuple

    def convert(self, *field_arrays):
        assert len(field_arrays) > 0, "Must have at least one input field"

        converted = []

        for field_values in zip(*field_arrays):
            tai_sec_delta = 0.0

            for unit, offset_raw in zip(self._units, field_values):
                offset_raw = float(offset_raw)

                if unit == "days":
                    tai_sec_delta += offset_raw*24*60*60
                elif unit == "hours":
                    tai_sec_delta += offset_raw*60*60
                elif unit == "minutes":
                    tai_sec_delta += offset_raw*60
                elif unit == "seconds":
                    tai_sec_delta += offset_raw
                elif unit == "milliseconds":
                    tai_sec_delta += offset_raw / self._MILLISECONDS_PER_SECOND
                elif unit == "microseconds":
                    tai_sec_delta += offset_raw / self._MICROSECONDS_PER_SECOND
                elif unit == "nanoseconds":
                    tai_sec_delta += offset_raw / self._NANOSECONDS_PER_SECOND

            converted_time = self._since + TimeDelta(tai_sec_delta, format="sec", scale="tai")
            converted.append(converted_time.utc.datetime) # still return UTC-scale Python datetimes

        converted = np.array(converted, dtype=object)

        return converted

def unpack_compression_settings(com_set_val: "bytes|int"):
    """
    Unpack image compression control register value.

    See `SciPacket.COMPRESSION_REG` for details.
    """
    if isinstance(com_set_val, bytes):
        assert len(com_set_val) == 2, f"Compression settings should be a 2-byte field, got {len(com_set_val)} bytes"
        compress_config = int.from_bytes(com_set_val, "big")
    elif isinstance(com_set_val, (int, np.integer)):
        assert com_set_val <= 0xFFFF, f"Compression settings should fit within 2 bytes, got \\x{com_set_val:X}"
        compress_config = int(com_set_val)
    else:
        raise TypeError
    settings_dict = {"SCALE": compress_config >> 8,
                     "RSVD": (compress_config >> 7) & 0b1,
                     "PMB_INIT": (compress_config >> 6) & 0b1,
                     "CMP_BYP": (compress_config >> 5) & 0b1,
                     "BSEL": (compress_config >> 3) & 0b11,
                     "SQRT": (compress_config >> 2) & 0b1,
                     "JPEG": (compress_config >> 1) & 0b1,
                     "TEST": compress_config & 0b1}
    return settings_dict


def unpack_acquisition_settings(acq_set_val: "bytes|int"):
    """
    Unpack CEB image acquisition register value.

    See `SciPacket.ACQUISITION_REG` for details.
    """
    if isinstance(acq_set_val, bytes):
        assert len(acq_set_val) == 4, f"Acquisition settings should be a 4-byte field, got {len(acq_set_val)} bytes"
        acquire_config = int.from_bytes(acq_set_val, "big")
    elif isinstance(acq_set_val, (int, np.integer)):
        assert acq_set_val <= 0xFFFFFFFF, f"Acquisition settings should fit within 4 bytes, got \\x{acq_set_val:X}"
        acquire_config = int(acq_set_val)
    else:
        raise TypeError
    settings_dict = {"DELAY": acquire_config >> 24,
                     "IMG_NUM": (acquire_config >> 21) & 0b111,
                     "EXPOSURE": (acquire_config >> 8) & 0x1FFF,
                     "TABLE1": (acquire_config >> 4) & 0b1111,
                     "TABLE2": acquire_config & 0b1111}
    return settings_dict



def read_tlm_defs(path):
    tlm = pd.read_excel(path, sheet_name=None)
    for sheet in tlm.keys():
        tlm[sheet] = tlm[sheet].rename(columns={c: c.strip() for c in tlm[sheet].columns})
        if "Start Byte" in tlm[sheet].columns:
            tlm[sheet]["Bit"] = tlm[sheet]["Start Byte"]*8 + tlm[sheet]["Start Bit"]
    apids = tlm["Overview"].dropna().copy()
    apids.index = [int(x.split("x")[1], 16) for x in apids["APID"]]
    apids.columns = ["Name", "APID", "Size_bytes", "Description", "Size_words", "Size_remainder"]
    apids.loc[:, "Size_bytes"] = apids["Size_bytes"].astype(int)
    return apids, tlm

def get_ccsds_data_type(sheet_type, data_size):
    if data_size > 64:
        return "fill"
    if sheet_type[0] == "F":
        return "float"
    if sheet_type[0] == "I":
        return "int"
    if sheet_type[0] == "U":
        return "uint"
    return "fill"

def create_packet_definitions(tlm, parse_expanding_fields=True):
    defs = {}
    for packet_name in FIXED_PACKETS:
        fields = []
        for i, row in tlm[packet_name].iterrows():
            if i > 6:  # CCSDSPy doesn't need the primary header, but it's in the .xls file, so we skip
                fields.append(PacketField(name=row["Mnemonic"],
                                          data_type=get_ccsds_data_type(row["Type"], row["Data Size"]),
                                          bit_length=row["Data Size"]))
        pkt = ccsdspy.FixedLength(fields)

        pkt.add_converted_field(
            (f"{packet_name}_HDR_SEC", f"{packet_name}_HDR_USEC"),
            "timestamp",
            TaiDatetimeConverter(
                since=SC_TIME_EPOCH,
                units=("seconds", "microseconds"),
            ),
        )

        if packet_name=="ENG_LED":
            # LED packets have extra times... so we'll just convert them here
            pkt.add_converted_field(
                ("LED_PLS_START_SEC", "LED_PLS_START_USEC"),
                "led_start_time",
                TaiDatetimeConverter(
                    since=SC_TIME_EPOCH,
                    units=("seconds", "microseconds"),
                ),
            )
            pkt.add_converted_field(
                ("LED_PLS_END_SEC", "LED_PLS_END_USEC"),
                "led_end_time",
                TaiDatetimeConverter(
                    since=SC_TIME_EPOCH,
                    units=("seconds", "microseconds"),
                ),
            )

        defs[packet_name] = pkt

    for packet_name in VARIABLE_PACKETS:
        fields = []
        num_fields = len(tlm[packet_name])
        for i, row in tlm[packet_name].iterrows():
            if i > 6 and i != num_fields - 1:  # the expanding packet is assumed to be last
                fields.append(PacketField(name=row["Mnemonic"],
                                          data_type=get_ccsds_data_type(row["Type"], row["Data Size"]),
                                          bit_length=row["Data Size"]))
            elif i == num_fields - 1 and parse_expanding_fields:
                fields.append(PacketArray(name=row["Mnemonic"],
                                          data_type="uint",
                                          bit_length=8,
                                          array_shape="expand"))
        pkt = ccsdspy.VariableLength(fields)

        pkt.add_converted_field(
            (f"{packet_name}_HDR_SEC", f"{packet_name}_HDR_USEC"),
            "timestamp",
            TaiDatetimeConverter(
                since=SC_TIME_EPOCH,
                units=("seconds", "microseconds"),
            ),
        )

        defs[packet_name] = pkt
    return defs

@task(cache_policy=NO_CACHE)
def detect_new_tlm_files(pipeline_config: dict, session=None) -> list[str]:
    session = Session(engine)

    tlm_directory = pipeline_config["tlm_directory"]
    found_tlm_files = list(glob(os.path.join(tlm_directory, "**/*.tlm"), recursive=True))

    # drop all files before the 'tlm_start_date'
    if "tlm_start_date" in pipeline_config:
        tlm_start_date = parse_datetime_str(pipeline_config["tlm_start_date"])
        found_tlm_file_dates = [datetime.strptime("_".join(os.path.basename(path).split("_")[3:-1]),
                                                  "%Y_%j_%H_%M")
                                for path in found_tlm_files]
        found_tlm_files = [path for path, date in zip(found_tlm_files, found_tlm_file_dates)
                               if date >= tlm_start_date]
    found_tlm_files = set(found_tlm_files)
    database_tlm_files = set([p[0] for p in session.query(TLMFiles.path).distinct().all()])

    return sorted(list(found_tlm_files - database_tlm_files))

def ingest_tlm_file(path: str,
                    defs: dict[str, ccsdspy.VariableLength | ccsdspy.FixedLength],
                    apid_name2num: dict[str, int]):
    session = Session(engine)

    tlm_db_entry = TLMFiles(
        path=path,
        successful=False,
        num_attempts=0,
        last_attempt=datetime.now(UTC),
    )
    session.add(tlm_db_entry)
    session.commit()

    parsed = TLMLoader(path, defs, apid_name2num).load()
    success = parsed is not None

    if success:
        for packet_name in parsed:
            sql_db_table = PACKETNAME2SQL[packet_name]
            num_packets = len(parsed[packet_name]["CCSDS_APID"])
            packet_numbers_used = list(range(0, num_packets, PACKET_CADENCE.get(packet_name, 1)))
            pkts = {i: {} for i in packet_numbers_used}
            try:
                for packet_num in packet_numbers_used:
                    pkts[packet_num]["packet_index"] = packet_num
                    pkts[packet_num]["tlm_id"] = tlm_db_entry.tlm_id
                    pkts[packet_num]["ccsds_sequence_count"] = parsed[packet_name]["CCSDS_SEQUENCE_COUNT"][packet_num]
                    pkts[packet_num]["ccsds_packet_length"] = parsed[packet_name]["CCSDS_PACKET_LENGTH"][packet_num]
                    pkts[packet_num]["timestamp"] = parsed[packet_name]["timestamp"][packet_num]
                    pkts[packet_num]["spacecraft_id"] = parsed[packet_name][f"{packet_name}_HDR_SCID"][packet_num]

                    # now we set special keywords used only in specific tables
                    if packet_name == "SCI_XFI":
                        pkts[packet_num]["is_used"] = False
                        pkts[packet_num]["flash_block"] = parsed[packet_name]["SCI_XFI_HDR_FLASH_BLOCK"][packet_num]
                        pkts[packet_num]["compression_settings"] = parsed[packet_name]["SCI_XFI_HDR_COM_SET"][packet_num]
                        pkts[packet_num]["acquisition_settings"] = parsed[packet_name]["SCI_XFI_HDR_ACQ_SET"][packet_num]
                        pkts[packet_num]["packet_group"] = parsed[packet_name]["SCI_XFI_HDR_IMG_PKT_GRP"][packet_num]
                    elif packet_name == "ENG_LED":
                        pkts[packet_num]["led_start_time"] = parsed[packet_name]["led_start_time"][packet_num]
                        pkts[packet_num]["led_end_time"] = parsed[packet_name]["led_end_time"][packet_num]

                session.execute(
                    sql_db_table.__table__.insert(),
                    list(pkts.values()),
                )
                session.commit()
            except:  # noqa: E722
                success = False
                session.rollback()

    tlm_db_entry.successful = success
    tlm_db_entry.num_attempts += 1
    tlm_db_entry.last_attempt = datetime.now(UTC)
    session.commit()
    session.close()

@task
def unpack_n_bit_values(packed: bytes, byteorder: str, n_bits=19) -> np.ndarray:
    logger = get_run_logger()
    if n_bits in (8, 16, 32, 64):
        trailing = len(packed)%(n_bits//8)
        if trailing:
            logger.debug(f"Truncating {trailing} extra bytes")
            packed = packed[:-trailing]
        return np.frombuffer(packed, dtype=np.dtype(f"u{n_bits//8}").newbyteorder(byteorder))
    bit_length = len(packed)*8
    bytes_as_ints = np.frombuffer(packed, "u1")
    results = []
    for bit in range(0, bit_length, n_bits):
        encompassing_bytes = bytes_as_ints[bit//8:-((bit+n_bits)//-8)]
        # "ceil" equivalent of a//b is -(-a//b), because of
        # http://python-history.blogspot.com/2010/08/why-pythons-integer-division-floors.html
        if len(encompassing_bytes)*8 < n_bits:
            logger.debug(f"Terminating at bit {bit} because there are only {len(encompassing_bytes)*8}"
                      f" bits left, which is not enough to make a {n_bits}-bit value.")
            break
        bit_within_byte = bit % 8
        bytes_value = 0
        if byteorder in ("little", "<"):
            bytes_value = int.from_bytes(encompassing_bytes, "little")
            bits_value = (bytes_value >> bit_within_byte) & (2**n_bits - 1)
        elif byteorder in ("big", ">"):
            extra_bits_to_right = len(encompassing_bytes)*8 - (bit_within_byte+n_bits)
            bytes_value = int.from_bytes(encompassing_bytes, "big")
            bits_value = (bytes_value >> extra_bits_to_right) & (2**n_bits - 1)
        else:
            raise ValueError("`byteorder` must be either 'little' or 'big'")
        results.append(bits_value)
    return np.asanyarray(results)

def organize_lz_fits_keywords(lz_packet_db, lz_packet):
    def temperature_formula(value):
        return -7.19959E-11*(value**3)+1.74252E-06*(value**2)+(0.067873*value)-239.6134821
    return {
        "LZTIME": lz_packet_db.timestamp.isoformat(),
        "CCDTEMP": temperature_formula(int(lz_packet["LZ_P1_P01_NFI_DET_PRI__WFI_DET_PRI"])),
        "ICMTEMP": temperature_formula(int(lz_packet["LZ_P1_P02_NFI_ICM_PRI__WFI_ICM_PRI"])),
        "FORTEMP": temperature_formula(int(lz_packet["LZ_P1_P03_NFI_BAFFWD_PY__WFI_OLA_PRI"])),
        "AFTTEMP": temperature_formula(int(lz_packet["LZ_P1_P04_NFI_BAFAFT_PZ__WFI_CLAM_PRI"])),
        "PFWTEMP": temperature_formula(int(lz_packet["LZ_P1_P05_NFI_PFW_MOT__WFI_PFW_MOT"])),
        "DOORTEMP": temperature_formula(int(lz_packet["LZ_P1_P06_NFI_HOPA__WFI_RAD_CEN"])),
        "CEBTEMP": temperature_formula(int(lz_packet["LZ_P1_P07_CEB_BASE_PRI"])),
        "HOUSTEMP": temperature_formula(int(lz_packet["LZ_P1_P08_STM_ELEC__WFI_CAM_MX"])),
        "FINGTEMP": temperature_formula(int(lz_packet["LZ_P1_P09_STM_DET_PRI__WFI_COLDF_PZ"])),
        "FPGATEMP": lz_packet["LZ_XTS_TEMP_FPGA"],
    }

def organize_pfw_fits_keywords(pfw_packet_db, pfw_packet):
    return {
        "PFWTIME": pfw_packet_db.timestamp.isoformat(),
        "PFWSTAT": pfw_packet["PFW_STATUS"],
        "STEPCALC": pfw_packet["STEP_CALC"],
        "CMDSTEPS": pfw_packet["LAST_CMD_N_STEPS"],
        "HOMEOVRD": pfw_packet["HOME_POSITION_OVRD"],
        "POSCURR": pfw_packet["POSITION_CURR"],
        "POSCMD": pfw_packet["POSITION_CMD"],
        "POSRAW": pfw_packet["RESOLVER_POS_RAW"],
        "POSRAW2": pfw_packet["RESOLVER_POS_CORR"],
        "READCNT": pfw_packet["RESOLVER_READ_CNT"],
        "LMNSTEP": pfw_packet["LAST_MOVE_N_STEPS"],
        "LMTIME": pfw_packet["LAST_MOVE_EXECUTION_TIME"],
        "LTSTEP": pfw_packet["LIFETIME_STEPS_TAKEN"],
        "LTTIME": pfw_packet["LIFETIME_EXECUTION_TIME"],
        "FSMSTAT": pfw_packet["FSM_CTRL_STATE"],
        "READSTAT": pfw_packet["READ_SUB_STATE"],
        "MOVSTAT": pfw_packet["MOVE_SUB_STATE"],
        "HOMESTAT": pfw_packet["HOME_SUB_STATE"],
        "HOMEPOS": pfw_packet["HOME_POSITION"],
        "RESSEL": pfw_packet["RESOLVER_SELECT"],
        "RESTOLH": pfw_packet["RESOLVER_TOLERANCE_HOME"],
        "RESTOLC": pfw_packet["RESOLVER_TOLERANCE_CURR"],
        "STEPSEL": pfw_packet["STEPPER_SELECT"],
        "STEPDLY": pfw_packet["STEPPER_RATE_DELAY"],
        "STEPRATE": pfw_packet["STEPPER_RATE"],
        "SHORTMV": pfw_packet["SHORT_MOVE_SETTLING_TIME_MS"],
        "LONGMV": pfw_packet["LONG_MOVE_SETTLING_TIME_MS"],
        "PFWOFF1": pfw_packet["PRIMARY_STEP_OFFSET_1"],
        "PFWOFF2": pfw_packet["PRIMARY_STEP_OFFSET_2"],
        "PFWOFF3": pfw_packet["PRIMARY_STEP_OFFSET_3"],
        "PFWOFF4": pfw_packet["PRIMARY_STEP_OFFSET_4"],
        "PFWOFF5": pfw_packet["PRIMARY_STEP_OFFSET_5"],
        "RPFWOFF1": pfw_packet["REDUNDANT_STEP_OFFSET_1"],
        "RPFWOFF2": pfw_packet["REDUNDANT_STEP_OFFSET_2"],
        "RPFWOFF3": pfw_packet["REDUNDANT_STEP_OFFSET_3"],
        "RPFWOFF4": pfw_packet["REDUNDANT_STEP_OFFSET_4"],
        "RPFWOFF5": pfw_packet["REDUNDANT_STEP_OFFSET_5"],
        "PFWPOS1": pfw_packet["PRIMARY_RESOLVER_POSITION_1"],
        "PFWPOS2": pfw_packet["PRIMARY_RESOLVER_POSITION_2"],
        "PFWPOS3": pfw_packet["PRIMARY_RESOLVER_POSITION_3"],
        "PFWPOS4": pfw_packet["PRIMARY_RESOLVER_POSITION_4"],
        "PFWPOS5": pfw_packet["PRIMARY_RESOLVER_POSITION_5"],
        "RPFWPOS1": pfw_packet["REDUNDANT_RESOLVER_POSITION_1"],
        "RPFWPOS2": pfw_packet["REDUNDANT_RESOLVER_POSITION_2"],
        "RPFWPOS3": pfw_packet["REDUNDANT_RESOLVER_POSITION_3"],
        "RPFWPOS4": pfw_packet["REDUNDANT_RESOLVER_POSITION_4"],
        "RPFWPOS5": pfw_packet["REDUNDANT_RESOLVER_POSITION_5"],
    }

def organize_led_fits_keywords(led_packet_db, led_packet):
    return {
        "LEDTIME": led_packet_db.timestamp.isoformat(),
        "LED1STAT": led_packet["LED1_ACTIVE_STATE"],
        "LEDPLSN": led_packet["LED_CFG_NUM_PLS"],
        "LED2STAT": led_packet["LED2_ACTIVE_STATE"],
        "LEDPLSD": led_packet["LED_CFG_PLS_DLY"],
        "LEDPLSW": led_packet["LED_CFG_PLS_WIDTH"],
    }


def organize_ceb_fits_keywords(ceb_packet_db, ceb_packet):
    return {
        "CEBTIME": ceb_packet_db.timestamp.isoformat(),
        "CEBSTAT": ceb_packet["CEB_STATUS_REG"],
        "CEBWGS": ceb_packet["WGS_STATUS"],
        "CEBFIFO": ceb_packet["VIDEO_FIFO_STATUS"],
        "CEBBIAS1": ceb_packet["CCD_OUTPUT_DRAIN_BIAS"],
        "CEBBIAS2": ceb_packet["CCD_DUMP_DRAIN_BIAS"],
        "CEBBIAS3": ceb_packet["CCD_RESET_DRAIN_BIAS"],
        "CEBBIAS4": ceb_packet["CCD_TOP_GATE_BIAS"],
        "CEBBIAS5": ceb_packet["CCD_OUTPUT_GATE_BIAS"],
        "CEBVREF": ceb_packet["VREF_P2_5V1"],
        "CEBGND1": ceb_packet["GROUND1"],
        "CEBCONV1": ceb_packet["DCDC_CONV_P30V_OUT"],
        "CEBCONV2": ceb_packet["DCDC_CONV_P15V_OUT"],
        "CEBCONV3": ceb_packet["DCDC_CONV_P5V_OUT"],
        "BIASVREF": ceb_packet["VREF_BIAS"],
        "CEBGND2": ceb_packet["GROUND2"],
        "CEBSEDAC": ceb_packet["IPF_SBE_CNT"],
        "CEBMEDAC": ceb_packet["IPF_MBE_CNT"]}


def organize_spacecraft_position_keywords(observation_time, before_xact_db, before_xact):
    position = EarthLocation.from_geocentric(before_xact["GPS_POSITION_ECEF1"]*2E-5*u.km,
                                             before_xact["GPS_POSITION_ECEF2"]*2E-5*u.km,
                                             before_xact["GPS_POSITION_ECEF3"]*2E-5*u.km)

    location = EarthLocation.from_geodetic(position.geodetic.lon.deg,
                                           position.geodetic.lat.deg,
                                           position.geodetic.height.to(u.m).value)
    obstime = Time(observation_time)

    gcrs = GCRS(location.get_itrs(obstime).cartesian, obstime=obstime)
    hci = gcrs.transform_to(HeliocentricInertial(obstime=obstime)) # HCI (Heliocentric Inertial)
    hee = gcrs.transform_to(HeliocentricEarthEcliptic(obstime=obstime)) # (Heliocentric Earth Ecliptic)
    hae = gcrs.transform_to(HeliocentricMeanEcliptic(obstime=obstime)) # HAE (Heliocentric Aries Ecliptic)
    heq = gcrs.transform_to(HeliographicStonyhurst(obstime=obstime)) # HEQ (Heliocentric Earth Equatorial)
    carrington = gcrs.transform_to(HeliographicCarrington(obstime=obstime, observer="self"))

    return {
        "XACTTIME": before_xact_db.timestamp.isoformat(),
        "HCIX_OBS": hci.cartesian.x.to(u.m).value,
        "HCIY_OBS": hci.cartesian.y.to(u.m).value,
        "HCIZ_OBS": hci.cartesian.z.to(u.m).value,
        "HEEX_OBS": hee.cartesian.x.to(u.m).value,
        "HEEY_OBS": hee.cartesian.y.to(u.m).value,
        "HEEZ_OBS": hee.cartesian.z.to(u.m).value,
        "HAEX_OBS": hae.cartesian.x.to(u.m).value,
        "HAEY_OBS": hae.cartesian.y.to(u.m).value,
        "HAEZ_OBS": hae.cartesian.z.to(u.m).value,
        "HEQX_OBS": heq.cartesian.x.to(u.m).value,
        "HEQY_OBS": heq.cartesian.y.to(u.m).value,
        "HEQZ_OBS": heq.cartesian.z.to(u.m).value,
        "HGLT_OBS": heq.lat.deg,
        "HGLN_OBS": heq.lon.deg,
        "CRLT_OBS": carrington.lat.deg,
        "CRLN_OBS": carrington.lon.deg,
        "DSUN_OBS": sun.earth_distance(obstime).to(u.m).value,
        "GEOD_LAT": position.geodetic.lat.deg,
        "GEOD_LON": position.geodetic.lon.deg,
        "GEOD_ALT": position.geodetic.height.to(u.m).value,
    }

def organize_compression_and_acquisition_settings(compression_settings, acquisition_settings):
    return {"SCALE": float(compression_settings["SCALE"]),
            "PMB_INIT": compression_settings["PMB_INIT"],
            "CMP_BYP": compression_settings["CMP_BYP"],
            "BSEL": compression_settings["BSEL"],
            "ISSQRT": compression_settings["SQRT"],
            "WASJPEG": compression_settings["JPEG"],
            "ISTEST": compression_settings["TEST"],
            "DELAY": acquisition_settings["DELAY"],
            "IMGCOUNT": acquisition_settings["IMG_NUM"]+1,
            "EXPTIME": acquisition_settings["EXPOSURE"]/10.0 * (1+acquisition_settings["IMG_NUM"]),
            "TABLE1": acquisition_settings["TABLE1"],
            "TABLE2": acquisition_settings["TABLE2"]}

def organize_gain_info(spacecraft_id):
    match spacecraft_id:
        case 0x2F:
            gains = {"GAINBTM": 4.98,"GAINTOP": 4.92}
        case 0x10:
            gains = {"GAINBTM": 4.93, "GAINTOP": 4.90}
        case 0x2C:
            gains = {"GAINBTM": 4.90, "GAINTOP": 5.04}
        case 0xF9:
            gains = {"GAINBTM": 4.94, "GAINTOP": 4.89}
        case _:
            gains = {"GAINBTM": 4.9, "GAINTOP": 4.9}
    return gains


def decode_image_packets(img_packets, compression_settings):
    if compression_settings["JPEG"] and not compression_settings["CMP_BYP"]:
        byte_stream = img_packets.tobytes()
        if b"\xFF\xD8" not in byte_stream:
            raise ValueError("Missing start of image indicator in byte stream")
        if b"\xFF\xD9" not in byte_stream:
            raise ValueError("Missing end of image indicator in byte stream")

    # check_for_full_image(img_packets)
    if compression_settings["JPEG"]: # JPEG bit enabled (upper two pathways)
        if compression_settings["CMP_BYP"]: # skipped actual JPEG-ification
            pixel_values = unpack_n_bit_values(img_packets, byteorder=">", n_bits=16)
            # either 12-bit values, but placed into 16b words where the 4 MSb are 0000; or 16-bit truncated pixel values
        else: # data is in JPEG-LS format
            pixel_values: np.ndarray = pylibjpeg.decode(img_packets.tobytes())
    else:
        pixel_values = unpack_n_bit_values(img_packets, byteorder="<", n_bits=19)
    if pixel_values.max() < 2**16:
        pixel_values = pixel_values.astype(np.uint16)
    else:
        pixel_values = pixel_values.astype(np.uint32)

    num_vals = pixel_values.size
    width = 2176 if num_vals > 2048 * 2048 else 2048
    if num_vals % width == 0:
        return pixel_values.reshape((-1, width)).T
    return np.ravel(pixel_values)[:width*(num_vals//width)].reshape((-1, width)).T


def determine_file_type(polarizer_packet, pfw_is_out_of_date, led_info, image_shape) -> str:
    if led_info is not None:
        return "DY"
    if image_shape != (2048, 2048):
        return "OV"
    if pfw_is_out_of_date:
        return "PX"
    position = int(polarizer_packet["RESOLVER_POS_CORR"])
    reference_positions = np.array([polarizer_packet["PRIMARY_RESOLVER_POSITION_1"],
                                    polarizer_packet["PRIMARY_RESOLVER_POSITION_2"],
                                    polarizer_packet["PRIMARY_RESOLVER_POSITION_3"],
                                    polarizer_packet["PRIMARY_RESOLVER_POSITION_4"],
                                    polarizer_packet["PRIMARY_RESOLVER_POSITION_5"]], dtype=int)

    # NFI and WFI have polarizers installed in different orientations. Thus, we treat them separately.
    # WFI has M and P flipped with respect to NFI.
    if polarizer_packet["ENG_PFW_HDR_SCID"] == 47:  # nfi case
        label = NFI_PFW_POSITION_MAPPING[np.argmin(np.abs(reference_positions - position))]
    else:  # wfi case
        label = WFI_PFW_POSITION_MAPPING[np.argmin(np.abs(reference_positions - position))]
    return label

def get_metadata(first_image_packet,
                 image_shape,
                 session,
                 defs,
                 apid_name2num,
                 pfw_recency_requirement=3,
                 xact_recency_requirement=3) -> tuple[dict[str, Any], dict[str, Any]]:
    acquisition_settings  = unpack_acquisition_settings(first_image_packet.acquisition_settings)
    compression_settings  = unpack_compression_settings(first_image_packet.compression_settings)

    offset_for_clearing = timedelta(seconds=3.8)
    observation_time = first_image_packet.timestamp + offset_for_clearing
    spacecraft_id = first_image_packet.spacecraft_id
    exposure_time = acquisition_settings["EXPOSURE"]/10.0 * (1+acquisition_settings["IMG_NUM"])

    # get the XACT packet right before and right after the first image packet to determine position
    before_xact_db = (session.query(ENG_XACT)
                   .filter(ENG_XACT.spacecraft_id == spacecraft_id)
                   .filter(ENG_XACT.timestamp <= observation_time)
                   .order_by(ENG_XACT.timestamp.desc()).first())
    after_xact_db = (session.query(ENG_XACT)
                  .filter(ENG_XACT.spacecraft_id == spacecraft_id)
                  .filter(ENG_XACT.timestamp >= observation_time)
                  .order_by(ENG_XACT.timestamp.asc()).first())

    # get the PFW packet right before the observation
    best_pfw_db = (session.query(ENG_PFW)
                  .filter(ENG_PFW.spacecraft_id == spacecraft_id)
                  .filter(ENG_PFW.timestamp <= observation_time)
                  .order_by(ENG_PFW.timestamp.desc()).first())
    pfw_recency = abs((best_pfw_db.timestamp - observation_time).total_seconds())
    pfw_is_out_of_date = pfw_recency > pfw_recency_requirement

    # get the CEB packet right before the observation
    best_ceb_db = (session.query(ENG_CEB)
                  .filter(ENG_CEB.spacecraft_id == spacecraft_id)
                  .filter(ENG_CEB.timestamp < observation_time)
                  .order_by(ENG_CEB.timestamp.desc()).first())

    # get the LZ packet right before the observation
    best_lz_db = (session.query(ENG_LZ)
                  .filter(ENG_LZ.spacecraft_id == spacecraft_id)
                  .filter(ENG_LZ.timestamp < observation_time)
                  .order_by(ENG_LZ.timestamp.desc()).first())

    # get the LED packet that corresponds to this observation if one exists.
    # this is slightly different, we look for an LED packet with a start time and an end time that overlaps
    # with the observation... there is likely not one, so this will be None.
    # there are multiple possibilities of overlaps, so we check them all and then just take one
    best_led1 = (session.query(ENG_LED)
                .filter(ENG_LED.spacecraft_id == spacecraft_id)
                .filter(ENG_LED.led_start_time <= observation_time)
                .filter(ENG_LED.led_end_time >= observation_time + timedelta(seconds=exposure_time))
                .first())

    best_led2 = (session.query(ENG_LED)
                .filter(ENG_LED.spacecraft_id == spacecraft_id)
                .filter(ENG_LED.led_start_time <= observation_time)
                .filter(ENG_LED.led_end_time >= observation_time)
                .filter(ENG_LED.led_end_time <= observation_time + timedelta(seconds=exposure_time))
                .first())

    best_led3 = (session.query(ENG_LED)
                .filter(ENG_LED.spacecraft_id == spacecraft_id)
                .filter(ENG_LED.led_start_time >= observation_time)
                .filter(ENG_LED.led_start_time <= observation_time + timedelta(seconds=exposure_time))
                .filter(ENG_LED.led_end_time >= observation_time + timedelta(seconds=exposure_time))
                .first())

    best_led4 = (session.query(ENG_LED)
                .filter(ENG_LED.spacecraft_id == spacecraft_id)
                .filter(ENG_LED.led_start_time >= observation_time)
                .filter(ENG_LED.led_end_time <= observation_time + timedelta(seconds=exposure_time))
                .first())

    best_led_db = best_led1 or best_led2 or best_led3 or best_led4

    packet_references = [before_xact_db, after_xact_db, best_ceb_db, best_pfw_db, best_led_db, best_lz_db]
    needed_tlm_ids = set([pkt.tlm_id for pkt in packet_references if pkt is not None])
    tlm_id_to_tlm_path = {tlm_id: session.query(TLMFiles.path).where(TLMFiles.tlm_id == tlm_id).one().path
                          for tlm_id in needed_tlm_ids}
    loaded_tlm = {}
    for tlm_id, tlm_path in tlm_id_to_tlm_path.items():
        parsed = TLMLoader(tlm_path, defs, apid_name2num).load()
        loaded_tlm[tlm_id] = parsed

    before_xact = {key: loaded_tlm[before_xact_db.tlm_id]["ENG_XACT"][key][before_xact_db.packet_index]
                   for key in loaded_tlm[before_xact_db.tlm_id]["ENG_XACT"]}
    after_xact = {key: loaded_tlm[after_xact_db.tlm_id]["ENG_XACT"][key][after_xact_db.packet_index]
                  for key in loaded_tlm[after_xact_db.tlm_id]["ENG_XACT"]}
    best_pfw = {key: loaded_tlm[best_pfw_db.tlm_id]["ENG_PFW"][key][best_pfw_db.packet_index]
                for key in loaded_tlm[best_pfw_db.tlm_id]["ENG_PFW"]}

    before_quat = np.quaternion(before_xact["ATT_DET_Q_BODY_WRT_ECI4"] * 0.5E-10,
                                before_xact["ATT_DET_Q_BODY_WRT_ECI1"] * 0.5E-10,
                                before_xact["ATT_DET_Q_BODY_WRT_ECI2"] * 0.5E-10,
                                before_xact["ATT_DET_Q_BODY_WRT_ECI3"] * 0.5E-10)

    after_quat = np.quaternion(after_xact["ATT_DET_Q_BODY_WRT_ECI4"] * 0.5E-10,
                               after_xact["ATT_DET_Q_BODY_WRT_ECI1"] * 0.5E-10,
                               after_xact["ATT_DET_Q_BODY_WRT_ECI2"] * 0.5E-10,
                               after_xact["ATT_DET_Q_BODY_WRT_ECI3"] * 0.5E-10)

    interp_quat = quaternion.slerp(before_quat, after_quat,
                                   before_xact_db.timestamp.timestamp(), after_xact_db.timestamp.timestamp(),
                                   observation_time.timestamp())

    position_info = {"spacecraft_id": spacecraft_id,
                     "datetime": observation_time,
                     "interp_quat": interp_quat,
                     "PFW_POSITION_CURR": best_pfw["POSITION_CURR"]}

    # fill in all the FITS info
    fits_info = {"TYPECODE": determine_file_type(best_pfw,
                                                 pfw_is_out_of_date,
                                                 best_led_db,
                                                 image_shape)}

    fits_info |= organize_pfw_fits_keywords(best_pfw_db, best_pfw)

    fits_info |= organize_spacecraft_position_keywords(observation_time, before_xact_db, before_xact)

    if best_led_db is not None:
        best_led = {key: loaded_tlm[best_led_db.tlm_id]["ENG_LED"][key][best_led_db.packet_index]
                    for key in loaded_tlm[best_led_db.tlm_id]["ENG_LED"]}
        fits_info |= organize_led_fits_keywords(best_led_db, best_led)
        fits_info["LED_PCKT"] = 1
    else:
        fits_info["LED_PCKT"] = 0

    if best_ceb_db is not None:
        best_ceb = {key: loaded_tlm[best_ceb_db.tlm_id]["ENG_CEB"][key][best_ceb_db.packet_index]
                    for key in loaded_tlm[best_ceb_db.tlm_id]["ENG_CEB"]}
        fits_info |= organize_ceb_fits_keywords(best_ceb_db, best_ceb)

    if best_lz_db is not None:
        best_lz = {key: loaded_tlm[best_lz_db.tlm_id]["ENG_LZ"][key][best_lz_db.packet_index]
                    for key in loaded_tlm[best_lz_db.tlm_id]["ENG_LZ"]}
        fits_info |= organize_lz_fits_keywords(best_lz_db, best_lz)

    fits_info |= organize_compression_and_acquisition_settings(compression_settings, acquisition_settings)
    if spacecraft_id == 0x2F:
        fits_info["RAWBITS"] = 19
    else:
        fits_info["RAWBITS"] = 16

    if fits_info["ISSQRT"] == 0:
        fits_info["BUNIT"] = "DN"
        fits_info["COMPBITS"] = fits_info["RAWBITS"]
        fits_info["DSATVAL"] = 2**fits_info["RAWBITS"] - 1
        fits_info["DESCRPTN"] = "PUNCH Level-0 data, DN values in camera coordinates"
    else:
        fits_info["BUNIT"] = "sqrt(DN)"
        fits_info["COMPBITS"] = round((fits_info["RAWBITS"] + int(np.log2(fits_info["SCALE"]))) / 2, 2)
        fits_info["DSATVAL"] = 2**fits_info["COMPBITS"] - 1
        fits_info["DESCRPTN"] = "PUNCH Level-0 data, square-root encoded DN values in camera coordinates"

    fits_info |= organize_gain_info(spacecraft_id)

    exposure_time = float(fits_info["EXPTIME"])
    fits_info["COM_SET"] = first_image_packet.compression_settings
    fits_info["ACQ_SET"] = first_image_packet.acquisition_settings
    fits_info["DATE-BEG"] = observation_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    date_end = observation_time + timedelta(seconds=exposure_time)
    fits_info["DATE-END"] = date_end.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    date_avg =  observation_time + timedelta(seconds=exposure_time/2)
    fits_info["DATE-AVG"] = date_avg.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    fits_info["DATE-OBS"] = date_avg.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    fits_info["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    return position_info, fits_info


def eci_quaternion_to_ra_dec(q, obstime):
    """
    Convert an ECI quaternion to RA and Dec.

    Args:
        q: A numpy array representing the ECI quaternion (q0, q1, q2, q3).

    Returns:
        ra: Right Ascension in degrees.
        dec: Declination in degrees.

    """
    # Normalize the quaternion
    q = q / np.abs(q)

    w, x, y, z = q.w, q.x, q.y, q.z
    # Calculate the rotation matrix from the quaternion
    R = np.array([[1 - 2*((y**2) + (z**2)), 2*((x*y) - (z*w)), 2*((x*z) + (y*w))],
         [2*((x*y) + (z*w)), 1 - 2*((x**2) + (z**2)), 2*((y*z) - (x*w))],
         [2*((x*z) - (y*w)), 2*((y*z) + (x*w)), 1 - 2*((x**2) + (y**2))]])

    axis_eci = np.array([1, 0, 0])
    body = R @ axis_eci

    # Calculate RA and Dec from the rotated z-vector
    c = SkyCoord(body[0], body[1], body[2],
                 representation_type="cartesian",
                 unit="m",
                 obstime=obstime).fk5
    ra = c.ra.deg
    dec = c.dec.deg
    roll = np.arctan2(2 *((w * x) + (y * z)), 1 - 2*((x ** 2) + (y ** 2)))

    return ra, dec, roll

def form_preliminary_wcs(soc_spacecraft_id, metadata, plate_scale):
    """Create the preliminary WCS for punchbowl"""
    q = metadata["interp_quat"]

    # all WFIs have their bore sight roughly 25 degrees up from spacecraft
    # so we rotate the quaternions before figuring out the RA/DEC
    if soc_spacecraft_id != "4":
        BORESIGHT_ANGLE = np.deg2rad(-25)  # this number comes from Craig as an estimate

        factor = np.sin(BORESIGHT_ANGLE / 2)
        x, y, z = 0, 1, 0  # we rotate around the y-axis
        rotation_quaternion = np.quaternion(np.cos(BORESIGHT_ANGLE / 2),
                                            x * factor,
                                            y * factor,
                                            z * factor)
        q = q * rotation_quaternion
        q = q / q.abs()

    ra, dec, roll = eci_quaternion_to_ra_dec(q, metadata["datetime"])
    projection = "ARC" if soc_spacecraft_id == "4" else "AZP"
    celestial_wcs = WCS(naxis=2)
    celestial_wcs.wcs.crpix = (1024.5, 1024.5)
    celestial_wcs.wcs.crval = (ra, dec)
    celestial_wcs.wcs.cdelt = plate_scale, plate_scale
    celestial_wcs.wcs.pc = calculate_pc_matrix(roll, celestial_wcs.wcs.cdelt)
    if soc_spacecraft_id == "4":
        celestial_wcs.wcs.set_pv([(2, 1, 0.0)])  # TODO: makes sure this is reasonably set
    celestial_wcs.wcs.ctype = f"RA--{projection}", f"DEC-{projection}"
    celestial_wcs.wcs.cunit = "deg", "deg"
    return calculate_helio_wcs_from_celestial(celestial_wcs, Time(metadata["datetime"]), (2048, 2048))


def form_single_image_caller(args):
    return form_single_image(*args)


def form_single_image(spacecraft, t, defs, apid_name2num, pipeline_config, spacecraft_secrets, outlier_limits,
                      masks, processing_flow_id):
    session = Session(engine)

    replay_needs = []
    skip_image, skip_reason = False, ""
    image_packets_entries = (session.query(SCI_XFI)
                             .filter(and_(SCI_XFI.timestamp == t,
                                          SCI_XFI.spacecraft_id == spacecraft))
                             .all())

    # Determine all the relevant TLM files
    needed_tlm_ids = set([image_packet.tlm_id for image_packet in image_packets_entries])
    tlm_id_to_tlm_path = {tlm_id: session.query(TLMFiles.path).where(TLMFiles.tlm_id == tlm_id).one().path
                          for tlm_id in needed_tlm_ids}
    needed_tlm_paths = list(session.query(TLMFiles.path).where(TLMFiles.tlm_id.in_(needed_tlm_ids)).all())
    needed_tlm_paths = [p.path for p in needed_tlm_paths]

    # parse any TLM files
    tlm_contents = []
    for tlm_id, tlm_path in tlm_id_to_tlm_path.items():
        parsed_contents = TLMLoader(tlm_path, defs, apid_name2num).load()
        if parsed_contents is not None:
            tlm_contents.append(parsed_contents)
        else:
            skip_image = True
            skip_reason = "Could not load all needed TLM files"
            print(f"Could not load all needed TLM files for spacecraft {spacecraft}")

    if not skip_image:
        # we want to get the packet contents and order them so an image can be made
        # to order the packets in the correct order for de-commutation
        order_dict = {}
        packet_entry_mapping = {}
        for packet_entry in image_packets_entries:
            sequence_count = packet_entry.ccsds_sequence_count
            if sequence_count in order_dict:
                order_dict[sequence_count].append(packet_entry.id)
            else:
                order_dict[sequence_count] = [packet_entry.id]
            packet_entry_mapping[packet_entry.id] = packet_entry

        # sometimes there are replays, so there are repeated packets
        # we use the packet with the largest packet_id because it's most likely the newest
        ordered_image_content = []
        sequence_counter = []
        ordered_image_packet_entries = []
        try:
            for sequence_count in sorted(list(order_dict.keys())):
                best_packet = max(order_dict[sequence_count])
                packet_entry = packet_entry_mapping[best_packet]
                ordered_image_packet_entries.append(packet_entry)
                tlm_content_index = needed_tlm_paths.index(tlm_id_to_tlm_path[packet_entry.tlm_id])
                selected_tlm_contents = tlm_contents[tlm_content_index]
                ordered_image_content.append(
                    selected_tlm_contents["SCI_XFI"]["SCI_XFI_IMG_DATA"][packet_entry.packet_index])
                sequence_counter.append(
                    selected_tlm_contents["SCI_XFI"]["SCI_XFI_HDR_IMG_PKT_GRP"][packet_entry.packet_index])
            # we check that the packets are in order now... if they're not, we'll skip.
            # we know a packet sequence is in order if the difference in the pkt_grp is either 1 or 255.
            # 1 is the nominal case
            # 255 indicates the packets rolled over in the 8-bit counter
            sequence_counter_diff = np.diff(np.array(sequence_counter))
            if not np.all(np.isin(sequence_counter_diff, [1, 255])):
                skip_image = True
                skip_reason = "Packets are out of order"

                # if this is the case, then we need a replay. So we'll log that
                # we don't know if we're missing the first or last packets, so we'll just
                # request an extra flash block on both sides to ensure we get enough (hopefully)
                replay_needs.append({
                    "spacecraft": spacecraft,
                    "start_time": ordered_image_packet_entries[0].timestamp.isoformat(),
                    "start_block": ordered_image_packet_entries[0].flash_block - 1,
                    "replay_length": ordered_image_packet_entries[-1].flash_block
                                     - ordered_image_packet_entries[0].flash_block + 1 + 2,
                    "note": skip_reason})
        except Exception as e:
            skip_image = True
            skip_reason = f"Image could not find all packets, {e}"
            traceback.print_exc()

    # we'll finally try to decompress the image, if it fails, we cannot make the image, so we proceed
    if not skip_image:
        try:
            compression_settings = unpack_compression_settings(ordered_image_packet_entries[0].compression_settings)
            image = decode_image_packets(np.concatenate(ordered_image_content), compression_settings)
            if image.shape != (2048, 2048) and image.shape != (2176, 4192):
                skip_image = True
                skip_reason = f"Image is wrong shape. Found {image.shape}"
                replay_needs.append({
                    "spacecraft": spacecraft,
                    "start_time": ordered_image_packet_entries[0].timestamp.isoformat(),
                    "start_block": ordered_image_packet_entries[0].flash_block - 1,
                    "replay_length": ordered_image_packet_entries[-1].flash_block
                                     - ordered_image_packet_entries[0].flash_block + 1 + 2,
                    "note": skip_reason})
        except Exception as e:
            skip_image = True
            skip_reason = f"Image decoding failed {e}"
            replay_needs.append({
                "spacecraft": spacecraft,
                "start_time": ordered_image_packet_entries[0].timestamp.isoformat(),
                "start_block": ordered_image_packet_entries[0].flash_block - 1,
                "replay_length": ordered_image_packet_entries[-1].flash_block
                                 - ordered_image_packet_entries[0].flash_block + 1 + 2,
                "note": skip_reason})
            traceback.print_exc()

    # now that we have the image we're ready to collect the metadata and write it to file
    if not skip_image:
        try:
            # we need to work out the SOC spacecraft ID from the MOC spacecraft id
            moc_index = spacecraft_secrets["moc"].index(ordered_image_packet_entries[0].spacecraft_id)
            soc_spacecraft_id = spacecraft_secrets["soc"][moc_index]
            pfw_recency_requirement = pipeline_config["flows"]["level0"]["options"].get("pfw_recency_requirement",
                                                                                        np.inf)
            xact_recency_requirement = pipeline_config["flows"]["level0"]["options"].get("xact_recency_requirement",
                                                                                         np.inf)
            position_info, fits_info = get_metadata(ordered_image_packet_entries[0],
                                                    image.shape,
                                                    session,
                                                    defs,
                                                    apid_name2num,
                                                    pfw_recency_requirement=pfw_recency_requirement,
                                                    xact_recency_requirement=xact_recency_requirement)
            if fits_info["TYPECODE"] == "PX":
                raise MissingMetadataError("PFW is stale so will wait to process.")
            fits_info["FILEVRSN"] = pipeline_config["file_version"]
            fits_info["PIPEVRSN"] = punchbowl.__version__
            fits_info["NUM_PCKT"] = len(image_packets_entries)
            fits_info["PCKTBYTE"] = len(np.concatenate(ordered_image_content).tobytes())
            file_type = fits_info["TYPECODE"]

            preliminary_wcs = form_preliminary_wcs(
                str(soc_spacecraft_id),
                position_info,
                float(pipeline_config["plate_scale"][str(soc_spacecraft_id)]))

            # we're ready to pack this into an NDCube to write as a FITS file using punchbowl
            meta = NormalizedMetadata.load_template(file_type + str(soc_spacecraft_id), "0")
            for meta_key, meta_value in fits_info.items():
                meta[meta_key] = meta_value
            cube = NDCube(data=image, meta=meta, wcs=preliminary_wcs)
            cube.meta.provenance = [os.path.basename(p) for p in needed_tlm_paths]
            cube.meta.history.add_now("form_single_image", f"ran with punchpipe v{__version__}")

            punch_io._update_statistics(cube, modify_inplace=True)

            date_obs = parse_datetime_str(fits_info["DATE-OBS"])

            selected_limits = None
            for limit_observatory, limit_type, limit_date, limit_filename, limits in outlier_limits:
                if limit_observatory != str(soc_spacecraft_id):
                    continue
                if limit_type != file_type[1]:
                    continue
                if limit_date > date_obs:
                    continue
                selected_limits = limits
                cube.meta.history.add_now("form_single_image", f"Outlier detection with {limit_filename}")
                break
            if selected_limits is None:
                if len(outlier_limits) and file_type in ["CR", "PM", "PZ", "PP"]:
                    raise RuntimeError(f"Could not find outlier limits for {get_base_file_name(cube)}")
                is_outlier = False
            else:
                # to_fits_header populates CROTA
                is_outlier = not selected_limits.is_good(cube.meta.to_fits_header(cube.wcs, False))

            selected_mask = None
            for mask_observatory, mask_date, mask in masks:
                if mask_observatory != str(soc_spacecraft_id):
                    continue
                if mask_date > date_obs:
                    continue
                selected_mask = mask
                break
            if selected_mask is None:
                if len(masks):
                    raise RuntimeError(f"Could not find mask for {get_base_file_name(cube)}")
                bad_packets = False
            else:
                bad_packets = np.any(cube.data[~selected_mask])

            is_outlier = is_outlier or bad_packets

            meta["OUTLIER"] = int(is_outlier)
            meta["BADPKTS"] = int(bad_packets)

            # if we don't have bad packets we can make the image
            # if we have bad packets but the replay delay has been met, then we go ahead and make the image
            # otherwise (when we have bad packets and could get a replay), we just skip and will make the image later
            replay_delay = timedelta(days=pipeline_config["flows"]["level0"]["options"].get("days_to_wait_for_replay",
                                                                                            7))
            if not bad_packets or (bad_packets and datetime.now(UTC) - date_obs > replay_delay):
                l0_db_entry = File(level="0",
                                   polarization="C" if file_type[0] == "C" else file_type[1],
                                   file_type=file_type,
                                   observatory=str(soc_spacecraft_id),
                                   file_version=pipeline_config["file_version"],
                                   software_version=__version__,
                                   outlier=is_outlier,
                                   bad_packets=bad_packets,
                                   date_created=parse_datetime_str(fits_info["DATE"]).replace(tzinfo=UTC).astimezone(),
                                   date_obs=date_obs,
                                   date_beg=parse_datetime_str(fits_info["DATE-BEG"]),
                                   date_end=parse_datetime_str(fits_info["DATE-END"]),
                                   state="created",
                                   processing_flow=processing_flow_id,
                                   crota=cube.meta["CROTA"].value)

                # finally, time to write to file
                out_path = os.path.join(l0_db_entry.directory(pipeline_config["root"]),
                                        get_base_file_name(cube)) + ".fits"
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                write_ndcube_to_fits(cube, out_path, overwrite=False, skip_stats=True)
                session.add(l0_db_entry)
                session.commit()
            else:  # we skipped because there are bad packets and it's possible we'll get a replay
                skip_image = True
                skip_reason = "Waiting for replay"
        except MissingMetadataError:
            session.rollback()
            skip_image = True
            skip_reason = "PFW was stale so wait to make the image."
        except Exception as e:
            session.rollback()
            skip_image = True
            skip_reason = f"Could not make metadata and write image, {e}"
            trace = traceback.format_exc()
            skip_reason += "\n" + trace

    # go back and do some cleanup if we skipped the image
    if skip_image:
        now = datetime.now(UTC)
        for packet in image_packets_entries:
            packet.is_used = False
            packet.num_attempts = packet.num_attempts + 1 if packet.num_attempts is not None else 1
            packet.last_attempt = now
            packet.last_skip_reason = skip_reason
    else:
        now = datetime.now(UTC)
        for packet in image_packets_entries:
            packet.is_used = True
            packet.num_attempts = packet.num_attempts + 1 if packet.num_attempts is not None else 1
            packet.last_attempt = now
    session.commit()
    session.close()
    return replay_needs, not skip_image, skip_reason

@flow
def level0_form_images(pipeline_config, defs, apid_name2num, outlier_limits, masks, session, logger,
                       processing_flow_id):
    spacecraft_secrets = SpacecraftMapping.load("spacecraft-ids").mapping.get_secret_value()

    now = datetime.now(UTC)
    retry_days = float(pipeline_config["flows"]["level0"]["options"].get("retry_days", 3.0))
    retry_window_start = now - timedelta(days=retry_days)

    distinct_spacecraft = (session.query(SCI_XFI.spacecraft_id)
                           .filter(or_(~SCI_XFI.is_used, SCI_XFI.is_used.is_(None)))
                           .distinct()
                           .all())

    skip_count, success_count = 0, 0
    replay_needs = []

    image_inputs = []
    for spacecraft in distinct_spacecraft:
        distinct_times = (session.query(SCI_XFI.timestamp)
                          .filter(or_(~SCI_XFI.is_used, SCI_XFI.is_used.is_(None)))
                          .filter(SCI_XFI.spacecraft_id == spacecraft[0])
                          .filter(SCI_XFI.timestamp > retry_window_start)
                          .distinct()
                          .all())
        for t in distinct_times:
            image_inputs.append((spacecraft[0], t[0], defs, apid_name2num, pipeline_config, spacecraft_secrets,
                                 outlier_limits, masks, processing_flow_id))
    logger.info(f"Got {len(image_inputs)} images to try forming")

    try:
        num_workers = pipeline_config["flows"]["level0"]["options"]["num_workers"]
    except KeyError:
        num_workers = 4
        logger.warning(f"No num_workers defined, using {num_workers} workers")

    with multiprocessing.get_context("spawn").Pool(num_workers, initializer=initializer) as pool:
        skip_reasons = defaultdict(lambda: 0)
        for i, (new_replay_needs, successful_image, skip_reason) in enumerate(
                pool.imap(form_single_image_caller, image_inputs, chunksize=10)):
            replay_needs.extend(new_replay_needs)
            if successful_image:
                success_count += 1
            else:
                skip_reasons[skip_reason] += 1
                skip_count += 1
            if i % 1000 == 0:
                logger.info(f"Completed {i} / {len(image_inputs)} image formation attempts")

    history = PacketHistory(datetime=datetime.now(UTC),
                            num_images_succeeded=success_count,
                            num_images_failed=skip_count)
    session.add(history)
    session.commit()
    logger.info(f"SUCCESS={success_count}")
    logger.info(f"FAILURE={skip_count}")

    for reason in skip_reasons:
        logger.info(f"Skipped {skip_reasons[reason]} images for reason {reason}")

    # Split into multiple files and append updates instead of making a new file each time
    # We label not with the spacecraft telemetry ID but with the spelled out name
    all_replays = pd.DataFrame(replay_needs)
    for df_spacecraft in all_replays.spacecraft.unique():
        date_str = datetime.now(UTC).strftime("%Y_%j")
        spacecraft_secrets = SpacecraftMapping.load("spacecraft-ids").mapping.get_secret_value()
        try:
            moc_index = spacecraft_secrets["moc"].index(df_spacecraft)
            soc_spacecraft_id = spacecraft_secrets["soc"][moc_index]
        except:  # noqa: E722
            # we cannot find the spacecraft id and need to use an unknown indicator
            soc_spacecraft_id = 0
        file_spacecraft_id = {0: "UNKN", 1: "WFI01", 2: "WFI02", 3: "WFI03", 4: "NFI00"}[soc_spacecraft_id]
        df_path = os.path.join(pipeline_config["root"],
                               "REPLAY",
                               f"PUNCH_{file_spacecraft_id}_REPLAY_{date_str}.csv")
        new_entries = all_replays[all_replays.spacecraft == df_spacecraft]
        new_entries = new_entries.drop(columns=["spacecraft"])
        if os.path.exists(df_path):
            existing_table = pd.read_csv(df_path)
            new_table = pd.concat([existing_table, new_entries], ignore_index=True)
            new_table = new_table.drop_duplicates()
        else:
            new_table = new_entries
        os.makedirs(os.path.dirname(df_path), exist_ok=True)
        new_table.to_csv(df_path, index=False)
    session.close()

@flow(log_prints=True)
def level0_core_flow(pipeline_config: dict, skip_if_no_new_tlm: bool = True, limit_files: list[str] = None,
                     mask_files: list[str] = None, processing_flow_id=None):
    logger = get_run_logger()
    session = Session(engine)

    outlier_limits = []
    if limit_files is not None:
        for limit_file in limit_files:
            limits = LimitSet.from_file(limit_file)
            file_name = os.path.basename(limit_file)
            code = file_name.split("_")[2][1]
            obs = file_name.split("_")[2][2]
            date = datetime.strptime(file_name.split("_")[3], "%Y%m%d%H%M%S")
            outlier_limits.append((obs, code, date, file_name, limits))

    masks = []
    if mask_files is not None:
        for mask_file in mask_files:
            mask = load_mask_file(mask_file)
            filename = os.path.basename(mask_file)
            observatory = filename.split("_")[2][2]
            date = datetime.strptime(filename.split("_")[3], "%Y%m%d%H%M%S")
            masks.append((observatory, date, mask))

    tlm_xls_path = pipeline_config["tlm_xls_path"]
    logger.info(f"Using {tlm_xls_path}")
    apids, tlm = read_tlm_defs(tlm_xls_path)
    apid_name2num = {row["Name"]: int(row["APID"], base=16) for _, row in apids.iterrows()}
    defs = create_packet_definitions(tlm, parse_expanding_fields=True)

    new_tlm_files = detect_new_tlm_files(pipeline_config, session=session)
    logger.info(f"Found {len(new_tlm_files)} new TLM files")

    if new_tlm_files or not skip_if_no_new_tlm:
        logger.debug("Proceeding through files")
        tlm_ingest_inputs = []
        for i, path in enumerate(new_tlm_files):
            tlm_ingest_inputs.append([path, defs, apid_name2num])

        try:
            num_workers = pipeline_config["flows"]["level0"]["options"]["num_workers"]
        except KeyError:
            num_workers = 4
            logger.warning(f"No num_workers defined, using {num_workers} workers")

        with multiprocessing.get_context("spawn").Pool(num_workers, initializer=initializer) as pool:
            pool.starmap(ingest_tlm_file, tlm_ingest_inputs)

        level0_form_images(pipeline_config, defs, apid_name2num, outlier_limits, masks, session, logger,
                           processing_flow_id)
    session.close()

def get_outlier_limits_paths(session, reference_time):
    limit_files = (session.query(File)
                     .filter(File.file_type.like("L%"))
                     .filter(File.level == "0")
                     .where(File.date_obs <= reference_time)
                     .order_by(File.file_version.desc(), File.date_obs.desc()).all())
    limit_files = [limit_file.filename().replace(".fits", ".npz") for limit_file in limit_files]
    return limit_files

def get_mask_paths(session, reference_time):
    mask_files = (session.query(File)
                     .filter(File.file_type == "MS")
                     .filter(File.level == "1")
                     .where(File.date_obs <= reference_time)
                     .order_by(File.file_version.desc(), File.date_obs.desc()).all())
    mask_files = [m.filename().replace(".fits", ".bin") for m in mask_files]
    return mask_files

@task(cache_policy=NO_CACHE)
def level0_construct_flow_info(pipeline_config: dict, session, skip_if_no_new_tlm: bool = True):
    flow_type = "level0"
    state = "planned"
    creation_time = datetime.now()
    priority = pipeline_config["flows"][flow_type]["priority"]["initial"]
    limits = get_outlier_limits_paths(session, creation_time)
    mask_files = get_mask_paths(session, creation_time)

    call_data = json.dumps(
        {
            "pipeline_config": pipeline_config,
            "skip_if_no_new_tlm": skip_if_no_new_tlm,
            "limit_files": limits,
            "mask_files": mask_files,
        },
    )
    return Flow(
        flow_type=flow_type,
        flow_level="0",
        state=state,
        creation_time=creation_time,
        priority=priority,
        call_data=call_data,
    )


@flow
def level0_scheduler_flow(pipeline_config_path=None, session=None, reference_time=None, skip_if_no_new_tlm: bool=True):
    pipeline_config = load_pipeline_configuration(pipeline_config_path)
    logger = get_run_logger()

    if session is None:
        session = Session(engine)

    # We have a concurrency limit set for the L0 flow. If we schedule another one while there's one pending or
    # running, that one could be launched, but then it could be cancelled by Prefect and so its state never gets
    # progressed beyond 'launched'. That will still count as something running for the launcher and will bog down the
    # pipeline.
    flows = (session.query(Flow)
             .where(Flow.state.in_(["planned", "running", "launched"]))
             .where(Flow.flow_type == "level0")
             .all())
    if len(flows):
        logger.info("Not scheduling---there's already a pending/running flow in the DB")
        return

    new_flow = level0_construct_flow_info(pipeline_config, session, skip_if_no_new_tlm=skip_if_no_new_tlm)

    session.add(new_flow)
    session.commit()


@flow
def level0_process_flow(flow_id: int, pipeline_config_path=None , session=None):
    logger = get_run_logger()

    if session is None:
        session = Session(engine)

    pipeline_config = load_pipeline_configuration(pipeline_config_path)

    # fetch the appropriate flow db entry
    flow_db_entry = session.query(Flow).where(Flow.flow_id == flow_id).one()
    logger.info(f"Running on flow db entry with id={flow_db_entry.flow_id}.")

    # update the processing flow name with the flow run name from Prefect
    flow_run_context = get_run_context()
    flow_db_entry.flow_run_name = flow_run_context.flow_run.name
    flow_db_entry.flow_run_id = flow_run_context.flow_run.id
    flow_db_entry.state = "running"
    flow_db_entry.start_time = datetime.now(UTC)
    session.commit()

    # load the call data and launch the core flow
    flow_call_data = json.loads(flow_db_entry.call_data)
    logger.info(f"Running with {flow_call_data}")

    flow_call_data["limit_files"] = file_name_to_full_path(flow_call_data["limit_files"], pipeline_config["root"])
    flow_call_data["mask_files"] = file_name_to_full_path(flow_call_data["mask_files"], pipeline_config["root"])

    try:
        level0_core_flow(**flow_call_data, processing_flow_id=flow_db_entry.flow_id)
    except Exception as e:
        flow_db_entry.state = "failed"
        flow_db_entry.end_time = datetime.now(UTC)
        logger.info("Something's gone wrong - level0_core_flow failed")
        session.commit()
        raise e
    else:
        flow_db_entry.state = "completed"
        flow_db_entry.end_time = datetime.now(UTC)
        # Note: the file_db_entry gets updated above in the writing step because it could be created or blank
        session.commit()


def open_and_split_packet_file(path: str) -> dict[int, io.BytesIO]:
    with open(path, "rb") as mixed_file:
        stream_by_apid = split_by_apid(mixed_file)
    return stream_by_apid

def parse_telemetry_file(path, defs, apid_name2num):
    success = True
    contents = open_and_split_packet_file(path)
    parsed = {}
    for packet_name in defs:
        apid_num = apid_name2num[packet_name]
        if apid_num in contents:
            try:
                parsed[packet_name] = defs[packet_name].load(contents[apid_num], include_primary_header=True)
            except (ValueError, RuntimeError):
                print(f"Unable to parse telemetry file {packet_name}")
                success = False
    return parsed, success

def short_hash(data, length=8):
    """Generates a short hash of specified length using MD5 and base64 encoding."""
    hash_object = hashlib.md5(data.encode())
    digest = hash_object.digest()
    truncated_digest = digest[:length]
    return base64.urlsafe_b64encode(truncated_digest).decode("ascii")

class TLMLoader(LoaderABC[dict]):
    def __init__(self, path: str, defs, apid_name2num):
        self.path = path
        self.defs = defs
        self.apid_name2num = apid_name2num

    def gen_key(self) -> str:
        return short_hash(f"tlm-{os.path.basename(self.path)}", length=16)

    def src_repr(self) -> str:
        return self.path

    def load_from_disk(self):
        print(f"loading from disk {self.path}!")
        try:
            parsed, _ = parse_telemetry_file(self.path, self.defs, self.apid_name2num)
        except Exception:
            parsed = None
        return parsed

    def __repr__(self):
        return f"TLM({self.path})"


def wrap_if_appropriate(psf_path: str, defs, apid_name2num) -> str | Callable:
    if manager.caching_is_enabled():
        return TLMLoader(psf_path, defs, apid_name2num).load
    return psf_path
