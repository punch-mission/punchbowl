# import tempfile
#
# from prefect.testing.utilities import prefect_test_harness
#
# from punchbowl.data import PUNCHData
# from punchbowl.level2.flow import level2_core_flow
# from punchbowl.tests.test_data import sample_punchdata, sample_data_random
#

# def test_core_flow_runs_with_filenames(sample_punchdata):
#     with tempfile.TemporaryDirectory() as tmpdirname:
#         input_name = tmpdirname + "/test_input.fits"
#         output_name = tmpdirname + "/test_output.fits"
#         sample_punchdata(shape=(2048, 2048)).write(input_name)
#         with prefect_test_harness():
#             level1_core_flow(input_name, output_filename=output_name)
#         output = PUNCHData.from_fits(output_name)
#         assert isinstance(output, PUNCHData)
#         # todo: test more things


# def test_core_flow_runs_with_objects(sample_punchdata):
#     with prefect_test_harness():
#         sample_list = [sample_punchdata(shape=(2048, 2048)),
#                        sample_punchdata(shape=(2048, 2048)),
#                        sample_punchdata(shape=(2048, 2048)),
#                        sample_punchdata(shape=(2048, 2048)),
#                        sample_punchdata(shape=(2048, 2048))]
#         output = level2_core_flow(sample_list)
#     assert isinstance(output[0], PUNCHData)
#     # todo: test more things


# TODO: reactivate
