import pathlib

from ndcube import NDCube
from prefect.testing.utilities import prefect_test_harness

from punchbowl.level1.flow import level1_core_flow
from punchbowl.tests.test_data import sample_data_random, sample_punchdata, sample_punchdata_clear

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


def test_core_flow_runs_with_objects(sample_punchdata):
    """Simply tests that the core flow runs with objects"""
    with prefect_test_harness():
        output = level1_core_flow(sample_punchdata(shape=(2048, 2048)))
    assert isinstance(output[0], NDCube)


def test_core_flow_runs_with_objects_and_calibration_files(sample_punchdata_clear):
    quartic_coefficient_path = THIS_DIRECTORY / "data" / "test_quartic_coeffs.fits"
    vignetting_path = THIS_DIRECTORY / "data" / "PUNCH_L1_GR1_20240222163425.fits"

    with prefect_test_harness():
        output = level1_core_flow(sample_punchdata_clear(shape=(10, 10)),
                                  quartic_coefficient_path=quartic_coefficient_path,
                                  vignetting_function_path=vignetting_path)
    assert isinstance(output[0], NDCube)
