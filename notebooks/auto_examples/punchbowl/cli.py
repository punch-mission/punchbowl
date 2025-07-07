import argparse
from datetime import datetime

from astropy.wcs import WCS
from dateutil.parser import parse as parse_datetime_str
from ndcube import NDCube

from punchbowl.data.meta import NormalizedMetadata
from punchbowl.data.punch_io import get_base_file_name, write_ndcube_to_fits
from punchbowl.level1.vignette import generate_vignetting_calibration


def main() -> None:
    """Call main method for CLI."""
    parser = argparse.ArgumentParser(prog="punchbowl")
    subparsers = parser.add_subparsers(dest="command")

    create_parser = subparsers.add_parser("create", help="Create calibration products.")

    create_parser.add_argument("level", type=str, help="Product level to make.")
    create_parser.add_argument("code", type=str, help="Product code to make.")
    create_parser.add_argument("spacecraft", type=str, help="Product spacecraft to make.")
    create_parser.add_argument("timestamp", type=str, help="DATE-OBS of output file.")
    create_parser.add_argument("file_version", type=str, help="Version of output calibration file.")
    create_parser.add_argument("input_file_path", type=str, help="Path to list of input files.")
    create_parser.add_argument("out_path", type=str, help="Path to output file.")

    args = parser.parse_args()

    if args.command == "create":
        args.timestamp = parse_datetime_str(args.timestamp)
        create_calibration(args.level, args.code, args.spacecraft, args.timestamp,
                           args.file_version, args.input_file_path, args.out_path)
    else:
        parser.print_help()


def create_calibration(level: str,
                       code: str,
                       spacecraft: str,
                       timestamp: datetime,
                       file_version: str,
                       input_list_path: str,
                       out_path: str) -> None:
    """
    Create calibration products.

    Parameters
    ----------
    level : str
        Product level to make.
    code : str
        Product code.
    spacecraft : str
        For WFIs, use 1, 2, or 3. For NFI, use 4.
    timestamp : datetime.datetime
        Output file's DATE-OBS.
    file_version : str
        Output file's VERSION.
    input_list_path : str
        Path to a list of filenames to use for generating this specific calibration product.
        For VG: the first entry is the Tappin data file. The second entry is the mask.
    out_path : str
        Directory to write calibration product to.

    Returns
    -------
    None

    """
    calibration_meta = NormalizedMetadata.load_template(f"{code}{spacecraft}", level)
    calibration_meta["DATE-OBS"] =  timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    calibration_meta["FILEVRSN"] = file_version

    with open(input_list_path) as input_list_file:
        input_list = input_list_file.readlines()

    match code:
        case "GR" | "GM" | "GZ" | "GP":
            calibration_data = generate_vignetting_calibration(input_list[0],
                                                                                input_list[1],
                                                                                spacecraft=spacecraft)
            calibration_cube = NDCube(data=calibration_data, wcs=WCS(naxis=2), meta=calibration_meta)
        case _:
            raise RuntimeError(f"Calibration pipeline not written for this code: {code}.")

    filename = f"{out_path}/{get_base_file_name(calibration_cube)}.fits"

    write_ndcube_to_fits(calibration_cube, filename=filename, overwrite=True,
                         write_hash=False, skip_wcs_conversion=True)
