import os
import pathlib

from ndcube import NDCube
from prefect.testing.utilities import prefect_test_harness

from punchbowl.data.io import write_ndcube_to_fits
from punchbowl.data.tests.test_io import sample_ndcube
from punchbowl.level1.flow import level1_core_flow

THIS_DIRECTORY = pathlib.Path(__file__).parent.resolve()

def test_core_flow_runs_with_filenames(sample_ndcube, tmpdir):
        input_name = os.path.join(tmpdir, "test_input.fits")
        output_name = os.path.join(tmpdir, "test_output.fits")
        write_ndcube_to_fits(sample_ndcube(shape=(10, 10), code="CR1", level="0"), input_name)
        quartic_coefficient_path = THIS_DIRECTORY / "data" / "test_quartic_coeffs.fits"
        with prefect_test_harness():
            output = level1_core_flow([input_name],
                                      quartic_coefficient_path=quartic_coefficient_path,
                                      output_filename=[output_name])
        assert isinstance(output[0], NDCube)
        assert os.path.exists(output_name[0])



def test_core_flow_runs_with_objects_and_calibration_files(sample_ndcube):
    cube = sample_ndcube(shape=(10, 10), code="CR1", level="0")
    quartic_coefficient_path = THIS_DIRECTORY / "data" / "test_quartic_coeffs.fits"

    with prefect_test_harness():
        output = level1_core_flow([cube],
                                  quartic_coefficient_path=quartic_coefficient_path)
    assert isinstance(output[0], NDCube)
