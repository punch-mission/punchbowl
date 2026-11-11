import os
from datetime import datetime

from prefect.logging import disable_run_logger
from prefect.testing.utilities import prefect_test_harness

from punchbowl.data.punch_io import write_ndcube_to_fits
from punchbowl.data.punchcube import PUNCHCube
from punchbowl.data.tests.test_punch_io import sample_ndcube
from punchbowl.data.wcs import load_trefoil_wcs
from punchbowl.levelq.flow import levelq_CQM_core_flow, levelq_CTM_core_flow, levelq_QAM_core_flow


def test_cqm_flow_runs_with_filenames(sample_ndcube, tmpdir):
    data_list = [sample_ndcube(shape=(10, 10), code=code, level="1") for code in ["QR1", "QR2", "QR3", "QR4"]]

    paths = []
    for i, cube in enumerate(data_list):
        path = os.path.join(tmpdir, f"test_input_{i}.fits")
        write_ndcube_to_fits(cube, path)
        paths.append(path)

    trefoil_wcs, _ = load_trefoil_wcs()
    with prefect_test_harness(), disable_run_logger():
        output = levelq_CQM_core_flow.fn(paths, trefoil_wcs=trefoil_wcs[::8, ::8], trefoil_shape=(512, 512))
    assert isinstance(output[0], PUNCHCube)
    assert output[0].meta["TYPECODE"].value == "CQ"

    assert output[0].meta["HAS_WFI1"].value == 1
    assert output[0].meta["HAS_WFI2"].value == 1
    assert output[0].meta["HAS_WFI3"].value == 1
    assert output[0].meta["HAS_NFI4"].value == 0


def test_QAM_metadata_not_empty(sample_ndcube, tmpdir):
    data_list = [sample_ndcube(shape=(10, 10), code="CTM", level="Q")]

    paths = []
    for i, cube in enumerate(data_list):
        path = os.path.join(tmpdir, f"test_input_{i}.fits")
        write_ndcube_to_fits(cube, path)
        paths.append(path)

    trefoil_wcs, _ = load_trefoil_wcs()
    with prefect_test_harness(), disable_run_logger():
        output = levelq_QAM_core_flow.fn(paths)
    assert isinstance(output[0], PUNCHCube)

    # Check if spacecraft location & environment headers are empty (error message should show value of keyword if fails)
    spacecraft_location_keys = ["GEOD_LAT","GEOD_LON","GEOD_ALT","HGLT_OBS","HGLN_OBS","CRLT_OBS","CRLN_OBS","DSUN_OBS","HEEX_OBS",
    "HEEY_OBS","HEEZ_OBS","HCIX_OBS","HCIY_OBS","HCIZ_OBS","HEQX_OBS","HEQY_OBS","HEQZ_OBS"]
    for keyword in spacecraft_location_keys:
        assert output[0].meta[keyword].value is not None and not "", f"{keyword}: {output[0].meta[keyword].value}"

    # TODO: The following is causing test failures
    # Check Moon-related headers # TODO: I'm unsure if this is just because of use of sample_ndcube
    moon_headers = ["MOONDIST","MOON_X","MOON_Y"]
    for keyword in moon_headers:
        assert output[0].meta[keyword].value is not None and not "", f"{keyword}: {output[0].meta[keyword].value}"

    # --Headers not included in "set_spacecraft_location_to_earth()" function as of 11/11/2026--
    spacecraft_location_keys2 = ["GEOX_OBS", "GEOY_OBS", "GEOZ_OBS", "GEOX_VOB", "GEOY_VOB", "GEOZ_VOB", "GSEX_OBS", "GSEY_OBS", "GSEZ_OBS",
    "GSEX_VOB", "GSEY_VOB", "GSEZ_VOB", "GEIX_OBS", "GEIY_OBS", "GEIZ_OBS", "GEIX_VOB", "GEIY_VOB", "GEIZ_VOB"]
    for keyword in spacecraft_location_keys2:
        assert output[0].meta[keyword].value is not None and not "", f"{keyword}: {output[0].meta[keyword].value}"
