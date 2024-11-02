import os
import argparse

import pandas as pd

from punchbowl.data import NormalizedMetadata
from punchbowl.data.meta import load_level_spec, load_spacecraft_def

LEVELS = ["0", "1", "2", "3", "L", "Q"]


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("-d", "--dir", help="path to the header yaml folder", required=True)
    return args.parse_args()


def check_all_files_exist(paths):
    for level in LEVELS:
        assert f"Level{level}.yaml" in paths, f"Level{level}.yaml not found"
    assert "spacecraft.yaml" in paths, "spacecraft.yaml not found"
    assert "omniheader.csv" in paths, "omniheader.csv not found"


def validate_omniheader(directory):
    omniheader = pd.read_csv(os.path.join(directory, "omniheader.csv"), na_filter=False)
    expected_columns = {"SECTION", "TYPE", "KEYWORD", "VALUE", "COMMENT", "DATATYPE", "NULLABLE", "MUTABLE", "DEFAULT"}
    assert set(omniheader.columns) == expected_columns, "Omniheader columns do not match expected"


def construct_all_product_headers(directory, level):
    level_path = os.path.join(directory, f"Level{level}.yaml")
    level_spec = load_level_spec(level_path)
    product_keys = list(level_spec["Products"].keys())
    crafts = load_spacecraft_def().keys()
    product_keys = sorted(list(set([pc.replace("?", craft) for craft in crafts for pc in product_keys])))
    for pc in product_keys:
        try:
            meta = NormalizedMetadata.load_template(pc, level)
        except Exception as e:
            assert False, f"failed to create {pc} for level {level} because: {e}"


if __name__ == "__main__":
    # provide directory with contents
    args = parse_args()

    check_all_files_exist(set(os.listdir(args.dir)))

    validate_omniheader(args.dir)

    for level in LEVELS:
        construct_all_product_headers(args.dir, level)

    print("No problems were found in the headers!")
