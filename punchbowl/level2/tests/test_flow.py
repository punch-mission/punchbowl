import os
import pathlib

import pytest
from prefect.logging import disable_run_logger
from prefect.testing.utilities import prefect_test_harness

from punchbowl.data.punch_io import write_ndcube_to_fits
from punchbowl.data.punchcube import PUNCHCube
from punchbowl.data.tests.test_punch_io import sample_ndcube
from punchbowl.data.wcs import load_trefoil_wcs
from punchbowl.level2.flow import POLARIZED_FILE_ORDER, level2_core_flow

THIS_DIRECTORY = pathlib.Path(__file__).parent.resolve()


def test_core_flow_runs_with_filenames(sample_ndcube, tmpdir):
    data_list = [sample_ndcube(shape=(10, 10), code=code, level="1") for code in POLARIZED_FILE_ORDER]
    paths = []
    for i, cube in enumerate(data_list):
        path = os.path.join(tmpdir, f"test_input_{i}.fits")
        write_ndcube_to_fits(cube, path)
        paths.append(path)
    voters = [[] for _ in data_list]

    trefoil_wcs, _ = load_trefoil_wcs()
    with prefect_test_harness(), disable_run_logger():
        output = level2_core_flow(paths, voters, trefoil_wcs=trefoil_wcs[::8, ::8], trefoil_shape=(512, 512))
    assert isinstance(output[0], PUNCHCube)
    assert output[0].meta["TYPECODE"].value == "PT"



def test_ctm_flow_runs_with_filenames(sample_ndcube, tmpdir):
    data_list = [sample_ndcube(shape=(10, 10), code=code, level="1") for code in ["CR1", "CR2", "CR3", "CR4"]]
    paths = []
    for i, cube in enumerate(data_list):
        path = os.path.join(tmpdir, f"test_input_{i}.fits")
        write_ndcube_to_fits(cube, path)
        paths.append(path)
    voters = [[] for _ in data_list]

    trefoil_wcs, _ = load_trefoil_wcs()
    with prefect_test_harness(), disable_run_logger():
        output = level2_core_flow(paths, voters, trefoil_wcs=trefoil_wcs[::8, ::8], trefoil_shape=(512, 512))
    assert isinstance(output[0], PUNCHCube)
    assert output[0].meta["TYPECODE"].value == "CT"
    assert output[0].meta["HAS_WFI1"].value == 1
    assert output[0].meta["HAS_WFI2"].value == 1
    assert output[0].meta["HAS_WFI3"].value == 1
    assert output[0].meta["HAS_NFI4"].value == 1


@pytest.mark.parametrize("drop_indices", [[], [1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
def test_core_flow_runs_with_objects_and_calibration_files(sample_ndcube, drop_indices, tmpdir):
    data_list = [sample_ndcube(shape=(100, 100), code=code, level="1")
                 for i, code in enumerate(POLARIZED_FILE_ORDER) if i not in drop_indices]

    trefoil_wcs, _ = load_trefoil_wcs()
    voters = [[] for _ in data_list]
    alphas_file = os.path.join(tmpdir, 'alphas.txt')
    with open(alphas_file, 'w') as f:
        f.write("""Prod Code,Alpha
CR1,0.97090
CR2,1.00000
CR3,1.07908
PM1,0.94474
PM2,1.00000
PM3,1.08709
PZ1,0.94794
PZ2,1.00000
PZ3,1.04966
PP1,1.00491
PP2,1.00000
PP3,1.09194
""")
    with prefect_test_harness(), disable_run_logger():
        output = level2_core_flow(data_list, voters, trefoil_wcs=trefoil_wcs[::8, ::8], trefoil_shape=(512, 512),
                                  polarized=True, trim_edges_px=1, alphas_file=alphas_file)
    assert isinstance(output[0], PUNCHCube)
    assert output[0].meta["TYPECODE"].value == "PT"

def test_spacecraft_location_metadata_not_empty(sample_ndcube,tmpdir):
    data_list = [sample_ndcube(shape=(10, 10), code=code, level="1") for code in POLARIZED_FILE_ORDER+["CR1", "CR2", "CR3", "CR4"]]
    paths = []
    for i, cube in enumerate(data_list):
        path = os.path.join(tmpdir, f"test_input_{i}.fits")
        write_ndcube_to_fits(cube, path)
        paths.append(path)
    voters = [[] for _ in data_list]

    trefoil_wcs, _ = load_trefoil_wcs()
    with prefect_test_harness(), disable_run_logger():
        output = level2_core_flow(paths, voters, trefoil_wcs=trefoil_wcs[::8, ::8], trefoil_shape=(512, 512))

    # Check if spacecraft location & environment headers are empty (error message should show value of keyword if fails)
    spacecraft_location_keys = ["GEOD_LAT","GEOD_LON","GEOD_ALT","HGLT_OBS","HGLN_OBS","CRLT_OBS","CRLN_OBS","DSUN_OBS","HEEX_OBS",
    "HEEY_OBS","HEEZ_OBS","HCIX_OBS","HCIY_OBS","HCIZ_OBS","HEQX_OBS","HEQY_OBS","HEQZ_OBS"]
    for keyword in spacecraft_location_keys:
        assert output[0].meta[keyword].value is not None and not "", f"{keyword}: {output[0].meta[keyword].value}"
    # Check Moon-related headers
    moon_headers = ["MOONDIST","MOON_X","MOON_Y"]
    for keyword in moon_headers:
        assert output[0].meta[keyword].value is not None and not "", f"{keyword}: {output[0].meta[keyword].value}"

    # TODO: Following is causing test failures
    # --Headers not included in "set_spacecraft_location_to_earth()" function as of 11/11/2026--
    spacecraft_location_keys2 = ["GEOX_OBS", "GEOY_OBS", "GEOZ_OBS", "GEOX_VOB", "GEOY_VOB", "GEOZ_VOB", "GSEX_OBS", "GSEY_OBS", "GSEZ_OBS",
    "GSEX_VOB", "GSEY_VOB", "GSEZ_VOB", "GEIX_OBS", "GEIY_OBS", "GEIZ_OBS", "GEIX_VOB", "GEIY_VOB", "GEIZ_VOB"]
    for keyword in spacecraft_location_keys2:
        assert output[0].meta[keyword].value is not None and not "", f"{keyword}: {output[0].meta[keyword].value}"
