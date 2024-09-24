import pathlib

from ndcube import NDCube
from prefect.testing.utilities import prefect_test_harness

from punchbowl.data.tests.test_io import sample_ndcube
from punchbowl.level1.flow import level1_core_flow

# def test_core_flow_runs_with_filenames(sample_punchdata):
#     with tempfile.TemporaryDirectory() as tmpdirname:
#         input_name = tmpdirname + "/test_input.fits"
#         output_name = tmpdirname + "/test_output.fits"
#         sample_punchdata(shape=(2048, 2048)).write(input_name)
#         with prefect_test_harness():
#             level1_core_flow(input_name, output_filename=output_name)
#         output = PUNCHData.from_fits(output_name)
#         assert isinstance(output, PUNCHData)

THIS_DIRECTORY = pathlib.Path(__file__).parent.resolve()


def test_core_flow_runs_with_objects_and_calibration_files(sample_ndcube):
    cube = sample_ndcube(shape=(10, 10), code="CR1", level="0")
    quartic_coefficient_path = THIS_DIRECTORY / "data" / "test_quartic_coeffs.fits"
    vignetting_path = THIS_DIRECTORY / "data" / "PUNCH_L1_GR1_20240222163425_v1.fits"

    with prefect_test_harness():
        output = level1_core_flow([cube],
                                  quartic_coefficient_path=quartic_coefficient_path,
                                  vignetting_function_path=vignetting_path)
    assert isinstance(output[0], NDCube)
