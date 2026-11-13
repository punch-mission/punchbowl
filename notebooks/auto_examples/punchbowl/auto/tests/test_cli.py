import os

from punchbowl.auto.cli import construct_flows_to_serve
from punchbowl.auto.speedster import load_pipeline_configuration

TEST_DIR = os.path.dirname(__file__)
pipeline_config_path = os.path.join(TEST_DIR, "../control/tests", "punchpipe_config.yaml")

def test_construct_flows_to_serve():
    config = load_pipeline_configuration(pipeline_config_path)
    flows = construct_flows_to_serve(pipeline_config_path)
    assert len(flows) == 2 * len(config['flows']) + len(config['control'])
