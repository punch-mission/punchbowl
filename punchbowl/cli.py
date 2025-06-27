import argparse

import numpy as np
from astropy.wcs import WCS
from ndcube import NDCube

from punchbowl.data.meta import NormalizedMetadata


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
                       shape: tuple[int, int] = (2048,2048),
                       ) -> NDCube:
    """Create calibration products."""
    m = NormalizedMetadata.load_template(f"{code}{spacecraft}", level)
    m["DATE-OBS"] = timestamp
    m["FILEVRSN"] = file_version

    return NDCube(data=np.zeros(shape), wcs=WCS(naxis=2), meta=m)
