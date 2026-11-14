import os
import sys
import pathlib
from datetime import datetime, timedelta

import pytest

from punchbowl.auto.control.cache_layer.f_corona import manager, wrap_if_appropriate
from punchbowl.conftest import prefect_test_fixture
from punchbowl.data.punch_io import write_ndcube_to_fits
from punchbowl.data.punchcube import PUNCHCube
from punchbowl.data.tests.test_punch_io import sample_ndcube
from punchbowl.level3.flow import level3_core_flow, level3_PIM_CIM_flow

THIS_DIRECTORY = pathlib.Path(__file__).parent.resolve()


@pytest.mark.parametrize('as_cubes', [True, False])
def test_PIM_flow_runs_with_filenames(sample_ndcube, tmpdir, prefect_test_fixture, as_cubes):
    data_list = [sample_ndcube(shape=(3, 10, 10), code=f"XP{obs + 1}", level="2") for obs in range(4)]
    for c in data_list:
        c.meta['CROPX2'] = 10
        c.meta['CROPY2'] = 10
        c.meta['FULXSIZE'] = 10
        c.meta['FULYSIZE'] = 10

    before_models = []
    for i, f in enumerate(data_list):
        before_f_corona_model = sample_ndcube(shape=(3, 10, 10), code="PF" + f.meta['OBSCODE'].value, level="3")
        before_f_corona_model_path = os.path.join(tmpdir, f"before_f_corona_{i}.fits")
        before_f_corona_model.meta['DATE-OBS'] = str(datetime(2024, 2, 22,
                                                              16, 0, 1)
                                                     - timedelta(hours=5))
        before_models.append(before_f_corona_model_path)
        write_ndcube_to_fits(before_f_corona_model, before_f_corona_model_path, write_hash=False, skip_stats=True)

    after_models = []
    for i, f in enumerate(data_list):
        after_f_corona_model = sample_ndcube(shape=(3, 10, 10), code="PF" + f.meta['OBSCODE'].value, level="3")
        after_f_corona_model_path = os.path.join(tmpdir, f"after_f_corona_{i}.fits")
        after_f_corona_model.meta['DATE-OBS'] = str(datetime(2024, 2, 22,
                                                             16, 0, 1)
                                                    + timedelta(hours=5))
        after_models.append(after_f_corona_model_path)
        write_ndcube_to_fits(after_f_corona_model, after_f_corona_model_path, write_hash=False, skip_stats=True)

    if not as_cubes:
        paths = []
        for i, cube in enumerate(data_list):
            path = os.path.join(tmpdir, f"test_input_{i}.fits")
            write_ndcube_to_fits(cube, path)
            paths.append(path)
        data_list = paths

    output = level3_PIM_CIM_flow(data_list,
                                 before_models,
                                 after_models,
                                 None)
    assert isinstance(output[0], PUNCHCube)
    assert output[0].meta['TYPECODE'].value == 'PI'
    assert output[0].meta['OBSCODE'].value == 'M'
    assert 'before_f_corona' in output[0].meta['CALFCOR1'].value
    assert 'after_f_corona' in output[0].meta['CALFCOR2'].value
    # Only the base filename should be there
    assert '/' not in output[0].meta['CALFCOR1'].value
    assert '/' not in output[0].meta['CALFCOR2'].value


@pytest.mark.parametrize('use_dataloaders',
                         [pytest.param(True,
                                       marks=pytest.mark.skipif(sys.version_info < (3, 13),
                                       reason="ShmPickleableArray requires Python 3.13 or higher")),
                          False])
def test_CIM_flow_clear_runs_with_filenames(sample_ndcube, tmpdir, prefect_test_fixture, use_dataloaders, monkeypatch):
    data_list = [sample_ndcube(shape=(10, 10), code=f"XR{obs + 1}", level="2") for obs in range(4)]
    for c in data_list:
        c.meta['CROPX2'] = 10
        c.meta['CROPY2'] = 10
        c.meta['FULXSIZE'] = 10
        c.meta['FULYSIZE'] = 10

    before_models = []
    for i, f in enumerate(data_list):
        before_f_corona_model = sample_ndcube(shape=(10, 10), code="CF" + f.meta['OBSCODE'].value, level="3")
        before_f_corona_model_path = os.path.join(tmpdir, f"before_f_corona_{i}.fits")
        before_f_corona_model.meta['DATE-OBS'] = str(datetime(2024, 2, 22,
                                                              16, 0, 1)
                                                     - timedelta(hours=5))
        before_models.append(before_f_corona_model_path)
        write_ndcube_to_fits(before_f_corona_model, before_f_corona_model_path, write_hash=False, skip_stats=True)

    after_models = []
    for i, f in enumerate(data_list):
        after_f_corona_model = sample_ndcube(shape=(10, 10), code="CF" + f.meta['OBSCODE'].value, level="3")
        after_f_corona_model_path = os.path.join(tmpdir, f"after_f_corona_{i}.fits")
        after_f_corona_model.meta['DATE-OBS'] = str(datetime(2024, 2, 22,
                                                             16, 0, 1)
                                                    + timedelta(hours=5))
        after_models.append(after_f_corona_model_path)
        write_ndcube_to_fits(after_f_corona_model, after_f_corona_model_path, write_hash=False, skip_stats=True)

    paths = []
    for i, cube in enumerate(data_list):
        path = os.path.join(tmpdir, f"test_input_{i}.fits")
        write_ndcube_to_fits(cube, path)
        paths.append(path)

    if use_dataloaders:
        with monkeypatch.context() as m:
            m.setattr(manager, 'caching_is_enabled', lambda: True)
            before_models = [wrap_if_appropriate(m) for m in before_models]
            after_models = [wrap_if_appropriate(m) for m in after_models]
        # Don't actually write anything to shared memory
        for m in before_models + after_models:
            m.try_caching = lambda *args, **kwargs: None

    output = level3_PIM_CIM_flow(data_list,
                                 before_models,
                                 after_models,
                                 None)
    assert isinstance(output[0], PUNCHCube)
    assert output[0].meta['TYPECODE'].value == 'CI'
    assert output[0].meta['OBSCODE'].value == 'M'
    assert 'before_f_corona' in output[0].meta['CALFCOR1'].value
    assert 'after_f_corona' in output[0].meta['CALFCOR2'].value
    # Only the base filename should be there
    assert '/' not in output[0].meta['CALFCOR1'].value
    assert '/' not in output[0].meta['CALFCOR2'].value


def test_level3_flow_PTM_runs_with_filenames(sample_ndcube, tmpdir, prefect_test_fixture):
    data_list = sample_ndcube(shape=(3, 10, 10), code="PIM", level="3")

    before_starfield_model = sample_ndcube(shape=(3, 10, 10), code="PSM", level="3")
    before_starfield_model_path = os.path.join(tmpdir, f"before_starfield.fits")
    before_starfield_model.meta['DATE-OBS'] = str(datetime(2024, 2, 22, 16, 0, 1)
                                                  - timedelta(hours=5))
    write_ndcube_to_fits(before_starfield_model, before_starfield_model_path, write_hash=False, skip_stats=True)

    after_starfield_model = sample_ndcube(shape=(3, 10, 10), code="PSM", level="3")
    after_starfield_model_path = os.path.join(tmpdir, f"after_starfield.fits")
    after_starfield_model.meta['DATE-OBS'] = str(datetime(2024, 2, 22, 16, 0, 1)
                                                 + timedelta(hours=5))
    write_ndcube_to_fits(after_starfield_model, after_starfield_model_path, write_hash=False, skip_stats=True)

    output = level3_core_flow([data_list],
                                 None,
                                 before_starfield_model_path,
                                 after_starfield_model_path,
                                 output_filename = None)
    assert isinstance(output[0], PUNCHCube)
    assert output[0].meta['TYPECODE'].value == 'PT'
    assert output[0].meta['OBSCODE'].value == 'M'


def test_level3_flow_CTM_runs_with_filenames(sample_ndcube, tmpdir, prefect_test_fixture):
    data_list = sample_ndcube(shape=(10, 10), code="CIM", level="3")

    before_starfield_model = sample_ndcube(shape=(10, 10), code="CSM", level="3")
    before_starfield_model_path = os.path.join(tmpdir, f"before_starfield.fits")
    before_starfield_model.meta['DATE-OBS'] = str(datetime(2024, 2, 22, 16, 0, 1)
                                                  - timedelta(hours=5))
    write_ndcube_to_fits(before_starfield_model, before_starfield_model_path, write_hash=False, skip_stats=True)

    after_starfield_model = sample_ndcube(shape=(10, 10), code="CSM", level="3")
    after_starfield_model_path = os.path.join(tmpdir, f"after_starfield.fits")
    after_starfield_model.meta['DATE-OBS'] = str(datetime(2024, 2, 22, 16, 0, 1)
                                                 + timedelta(hours=5))
    write_ndcube_to_fits(after_starfield_model, after_starfield_model_path, write_hash=False, skip_stats=True)

    output = level3_core_flow([data_list],
                                 None,
                                 before_starfield_model_path,
                                 after_starfield_model_path,
                                 output_filename = None)
    assert isinstance(output[0], PUNCHCube)
    assert output[0].meta['TYPECODE'].value == 'CT'
    assert output[0].meta['OBSCODE'].value == 'M'


def test_level3_metadata_not_empty(sample_ndcube,tmpdir,prefect_test_fixture):
    data_list = sample_ndcube(shape=(10, 10), code="CIM", level="3")

    before_starfield_model = sample_ndcube(shape=(10, 10), code="CSM", level="3")
    before_starfield_model_path = os.path.join(tmpdir, f"before_starfield.fits")
    before_starfield_model.meta['DATE-OBS'] = str(datetime(2024, 2, 22, 16, 0, 1)
                                                  - timedelta(hours=5))
    write_ndcube_to_fits(before_starfield_model, before_starfield_model_path, write_hash=False, skip_stats=True)

    after_starfield_model = sample_ndcube(shape=(10, 10), code="CSM", level="3")
    after_starfield_model_path = os.path.join(tmpdir, f"after_starfield.fits")
    after_starfield_model.meta['DATE-OBS'] = str(datetime(2024, 2, 22, 16, 0, 1)
                                                 + timedelta(hours=5))
    write_ndcube_to_fits(after_starfield_model, after_starfield_model_path, write_hash=False, skip_stats=True)

    output = level3_core_flow([data_list],
                                 None,
                                 before_starfield_model_path,
                                 after_starfield_model_path,
                                 output_filename = None)

    assert output[0].meta['CALFCOR1'] is not None and not "", "CALFCOR1 is empty"
    assert output[0].meta['CALFCOR2'] is not None and not "", "CALFCOR2 is empty"

    # Check if spacecraft location & environment headers are empty (error message should show value of keyword if fails)
    spacecraft_location_keys = ["GEOD_LAT","GEOD_LON","GEOD_ALT","GEOX_OBS","GEOY_OBS","GEOZ_OBS","HGLT_OBS","HGLN_OBS","CRLT_OBS","CRLN_OBS","DSUN_OBS","HEEX_OBS",
    "HEEY_OBS","HEEZ_OBS","HCIX_OBS","HCIY_OBS","HCIZ_OBS","HEQX_OBS","HEQY_OBS","HEQZ_OBS","GSEX_OBS", "GSEY_OBS", "GSEZ_OBS",
    "GSEX_VOB", "GSEY_VOB", "GSEZ_VOB", "GEIX_OBS", "GEIY_OBS", "GEIZ_OBS", "GEIX_VOB", "GEIY_VOB", "GEIZ_VOB"]
    for keyword in spacecraft_location_keys:
        assert output[0].meta[keyword].value is not None and not "", f"{keyword}: {output[0].meta[keyword].value}"

    # TODO: The following is causing test failures
    # Check Moon-related headers # TODO: I'm unsure if this is just because of use of sample_ndcube
    moon_headers = ["MOONDIST","MOON_X","MOON_Y"]
    for keyword in moon_headers:
        assert output[0].meta[keyword].value is not None and not "", f"{keyword}: {output[0].meta[keyword].value}"
