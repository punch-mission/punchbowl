from punchbowl.auto.control.util import load_pipeline_configuration


def test_config_merge(tmp_path):
    main_path = tmp_path / 'main_config.yaml'
    sub_path = tmp_path / 'sub_config.yaml'
    sub2_path = tmp_path / 'sub_config2.yaml'

    main_config = f"""
key_a: 1
key_b: 'string'
key_c: 'string'
includes:
  - {sub_path}
  - {sub2_path}
section1:
  key_1_a: 1
  key_1_b: 2
  key_1_c: 2
"""

    sub_config = """
key_z: 'z'
key_b: 'modified'
key_c: 'modified'
section1:
  key_1_z: 'z'
  key_1_b: 'modified'
  key_1_c: 'modified'
  section1a:
    key_1a_z: 'z'
section2:
  key_2_z: 'z'
  section2a:
    key_2a_z: 'z'
"""

    sub_config2 = """
key_q: 'q'
key_c: 'doubly modified'
section1:
  key_1_q: 'q'
  key_1_c: 'doubly modified'
section2:
  key_2_q: 'q'
  section2a:
    key_2a_q: 'q'
"""
    with open(main_path, 'w') as f:
        f.write(main_config)
    with open(sub_path, 'w') as f:
        f.write(sub_config)
    with open(sub2_path, 'w') as f:
        f.write(sub_config2)

    config = load_pipeline_configuration(main_path)
    assert config['key_a'] == 1
    assert config['key_b'] == 'modified'
    assert config['key_c'] == 'doubly modified'
    assert config['key_z'] == 'z'

    assert 'includes' not in config
    assert config['section1']['key_1_a'] == 1
    assert config['section1']['key_1_b'] == 'modified'
    assert config['section1']['key_1_c'] == 'doubly modified'
    assert config['section1']['key_1_q'] == 'q'
    assert config['section1']['key_1_z'] == 'z'

    assert config['section1']['section1a']['key_1a_z'] == 'z'

    assert config['section2']['key_2_q'] == 'q'
    assert config['section2']['key_2_z'] == 'z'

    assert config['section2']['section2a']['key_2a_q'] == 'q'
    assert config['section2']['section2a']['key_2a_z'] == 'z'
