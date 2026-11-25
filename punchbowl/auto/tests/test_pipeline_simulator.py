from datetime import datetime

import yaml
from PIL import CurImagePlugin
from yaml.loader import FullLoader

from punchbowl.auto.control.util import load_pipeline_configuration
from punchbowl.auto.pipeline_simulator import CAM, PAM, File, build_tree, mark_file_existence, plot_tree

TEST_CONFIG = """
flows:
  construct_stray_light:
    model_spacing_hours: 168 # 1 week
    t0: "2025-04-25 00:00:00"
    clear_min_files_per_half: 50
    clear_max_files_per_half: 50
    clear_file_stride: 1
    clear_max_hours_per_half: 200 # 5 days as the target, plus lots of extra time for data gaps
    clear_neighborhood_size: 9
    pol_min_files_per_half: 50
    pol_max_files_per_half: 50
    pol_file_stride: 2
    pol_max_hours_per_half: 200 # 5 days as the target, plus lots of extra time for data gaps
    pol_neighborhood_size: 9

  construct_dynamic_stray_light:
    model_spacing_hours: 120
    t0: "2025-04-25 00:00:00"
    min_files_per_half: 50
    max_files_per_half: 50
    max_hours_per_half: 180

  level2:
    accepted_phases: "5,6,7"

  construct_f_corona_background:
    model_spacing_hours: 96
    t0: "2025-06-10 00:00:00"
    clear_min_files_per_half: 50
    clear_max_files_per_half: 50
    clear_max_hours_per_half: 475 # 5 days, plus margin for missing files
    clear_file_stride: 3
    pol_min_files_per_half: 50
    pol_max_files_per_half: 50
    pol_max_hours_per_half: 475 # 5 days, plus margin for missing files
    pol_file_stride: 3
    nfi_min_files_per_half: 50
    nfi_max_files_per_half: 50
    nfi_max_hours_per_half: 600
    nfi_file_stride: 2
    options:

  level3_PAM:
    t0: "2025-04-25 00:00:00"

  construct_starfield_background:
    model_spacing_days: 7
    t0: "2025-06-10 00:00:00"
    clear_before_min_files: 50
    clear_after_min_files: 50
    clear_before_max_files: 50
    clear_after_max_files: 50
    pol_before_min_files: 50
    pol_after_min_files: 50
    pol_before_max_files: 50
    pol_after_max_files: 50
    days_before: 35
    days_after: 35
    clear_image_cadence: 15
    pol_image_cadence: 15

  level3_CAM:
    t0: "2025-04-25 00:00:00"
"""


def test_pipeline_simulator():
    config = yaml.load(TEST_CONFIG, Loader=FullLoader)

    File.set_pipeline_config(config)
    top = [PAM().get_file_closest_to(datetime(2025, 9, 1)), CAM().get_file_closest_to(datetime(2025, 9, 1))]
    tree_data = build_tree(top)
    plot_tree(tree_data)
