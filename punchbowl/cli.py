import argparse

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

    args = parser.parse_args()

    if args.command == "create":
        create_calibration(args.level, args.code)
    else:
        parser.print_help()


def create_calibration(level: str,
                       code: str,
                       spacecraft: str,
                       timestamp: str,
                       file_version: str,
                       outpath: str) -> NDCube:
    """Create calibration products."""
    calibration_meta = NormalizedMetadata.load_template(f"{code}{spacecraft}", level)
    calibration_meta["DATE-OBS"] = timestamp
    calibration_meta["FILEVRSN"] = file_version

    match code:
        case "VG":
            calibration_data, calibration_wcs = generate_vignetting_calibration(spacecraft=spacecraft,
                                                                                timestamp=timestamp)

    calibration_cube = NDCube(data=calibration_data, wcs=calibration_wcs, meta=calibration_meta)

    filename = f"{outpath}{get_base_file_name(calibration_cube)}.fits"

    write_ndcube_to_fits(calibration_cube, filename=filename, overwrite=True, write_hash=False)

    return None
