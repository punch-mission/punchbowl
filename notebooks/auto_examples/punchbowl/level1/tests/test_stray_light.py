import os
import pathlib

from ndcube import NDCube
from prefect.logging import disable_run_logger

from punchbowl.data import write_ndcube_to_fits
from punchbowl.data.tests.test_punch_io import sample_ndcube
from punchbowl.level1.stray_light import estimate_stray_light, remove_stray_light_task

THIS_DIRECTORY = pathlib.Path(__file__).parent.resolve()


def test_no_straylight_file(sample_ndcube) -> None:
    """
    An invalid vignetting file should be provided. Check that an error is raised.
    """

    sample_data = sample_ndcube(shape=(10, 10), code="CR1", level="0")
    straylight_before_filename = None
    straylight_after_filename = None

    with disable_run_logger():
        corrected_punchdata = remove_stray_light_task.fn(sample_data, straylight_before_filename, straylight_after_filename)
        assert isinstance(corrected_punchdata, NDCube)
        assert corrected_punchdata.meta.history[0].comment == 'Stray light correction skipped'


def test_estimate_stray_light_runs(tmpdir, sample_ndcube):
    data_list = [sample_ndcube(shape=(10, 10), code='XR1', level="1") for i in range(10)]

    paths = []
    for i, cube in enumerate(data_list):
        path = os.path.join(tmpdir, f"test_input_{i}.fits")
        write_ndcube_to_fits(cube, path)
        paths.append(path)

    with disable_run_logger():
        cube = estimate_stray_light.fn(paths, 3)

    assert cube[0].meta['TYPECODE'].value == 'SR'
    assert cube[0].meta['OBSCODE'].value == '1'
