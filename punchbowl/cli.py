import argparse


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

def create_calibration(level: str, code: str) -> None:
    """Create calibration product."""

if __name__ == "__main__":
    main()
