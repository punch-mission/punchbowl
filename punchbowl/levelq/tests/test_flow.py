import os
from datetime import datetime

from ndcube import NDCube
from prefect.logging import disable_run_logger
from prefect.testing.utilities import prefect_test_harness

from punchbowl.data.punch_io import write_ndcube_to_fits
from punchbowl.data.tests.test_punch_io import sample_ndcube
from punchbowl.data.wcs import load_trefoil_wcs
from punchbowl.levelq.flow import levelq_CQM_core_flow, levelq_CTM_core_flow


def test_ctm_flow_runs_with_filenames(sample_ndcube, tmpdir):
    input_image = sample_ndcube(shape=(10, 10), code='CQM', level="Q", date_obs='2025-01-03 00:00:00')
    input_image.meta['HAS_WFI1'].value = 1
    input_image.meta['HAS_WFI2'].value = 1
    input_image.meta['HAS_WFI3'].value = 1
    input_image.meta['HAS_NFI4'].value = 1
    fcorona_before = sample_ndcube(shape=(10, 10), code="CFM", level="Q", date_obs='2025-01-02 00:00:00')
    fcorona_after = sample_ndcube(shape=(10, 10), code="CFM", level="Q", date_obs='2025-01-04 00:00:00')

    paths = []
    for i, cube in enumerate([input_image, fcorona_before, fcorona_after]):
        path = os.path.join(tmpdir, f"test_input_{i}.fits")
        write_ndcube_to_fits(cube, path)
        paths.append(path)

    trefoil_wcs, _ = load_trefoil_wcs()
    with prefect_test_harness(), disable_run_logger():
        output = levelq_CTM_core_flow.fn([paths[0]],
                                         before_f_corona_model_path=paths[1],
                                         after_f_corona_model_path=paths[2]                                         )
    assert isinstance(output[0], NDCube)
    assert output[0].meta["TYPECODE"].value == "CT"

    assert output[0].meta["HAS_WFI1"].value == 1
    assert output[0].meta["HAS_WFI2"].value == 1
    assert output[0].meta["HAS_WFI3"].value == 1
    assert output[0].meta["HAS_NFI4"].value == 1


def test_cqm_flow_runs_with_filenames(sample_ndcube, tmpdir):
    data_list = [sample_ndcube(shape=(10, 10), code=code, level="1") for code in ["QR1", "QR2", "QR3"]]
    data_list.append(sample_ndcube(shape=(10, 10), code="CNN", level="Q"))

    paths = []
    for i, cube in enumerate(data_list):
        path = os.path.join(tmpdir, f"test_input_{i}.fits")
        write_ndcube_to_fits(cube, path)
        paths.append(path)

    trefoil_wcs, _ = load_trefoil_wcs()
    with prefect_test_harness(), disable_run_logger():
        output = levelq_CQM_core_flow.fn(paths, trefoil_wcs=trefoil_wcs[::8, ::8], trefoil_shape=(512, 512))
    assert isinstance(output[0], NDCube)
    assert output[0].meta["TYPECODE"].value == "CQ"

    assert output[0].meta["HAS_WFI1"].value == 1
    assert output[0].meta["HAS_WFI2"].value == 1
    assert output[0].meta["HAS_WFI3"].value == 1
    assert output[0].meta["HAS_NFI4"].value == 1
