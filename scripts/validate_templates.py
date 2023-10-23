from collections import OrderedDict
import yaml
import argparse
import os

import pandas as pd
import numpy as np

from punchbowl.data import History, MetaField, NormalizedMetadata


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("-d", "--dir", help="path to the data", required=True)
    return args.parse_args()


def check_all_files_exist(paths):
    assert "Level0.yaml" in paths, "Level0.yaml not found"
    assert "Level1.yaml" in paths, "Level1.yaml not found"
    assert "Level2.yaml" in paths, "Level2.yaml not found"
    assert "Level3.yaml" in paths, "Level3.yaml not found"
    assert "LevelL.yaml" in paths, "LevelL.yaml not found"
    assert "LevelQ.yaml" in paths, "LevelQ.yaml not found"
    assert "spacecraft.yaml" in paths, "spacecraft.yaml not found"
    assert "omniheader.csv" in paths, "omniheader.csv not found"


def validate_omniheader(directory):
    omniheader = pd.read_csv(os.path.join(directory, "omniheader.csv"), na_filter=False)
    expected_columns = {"SECTION", "TYPE", "KEYWORD", "VALUE", "COMMENT", "DATATYPE", "NULLABLE", "MUTABLE", "DEFAULT"}
    assert set(omniheader.columns) == expected_columns, "Omniheader columns do not match expected"
    # TODO: validate the values in each column match allowed, e.g. SECTION are only numbers


if __name__ == "__main__":
    # provide directory with contents
    args = parse_args()
    print(args.dir)

    check_all_files_exist(set(os.listdir(args.dir)))

    validate_omniheader(args.dir)

    for level in ["0", "1", "2", "3", "L", "Q"]:
        print(level)


