import json
from datetime import datetime, timedelta
from itertools import pairwise
from collections import defaultdict

from dateutil.parser import parse as parse_datetime_str
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from sqlalchemy import and_, func, or_, text
from sqlalchemy.orm import aliased

from punchbowl import __version__
from punchbowl.auto.control import cache_layer
from punchbowl.auto.control.db import File, FileRelationship, Flow
from punchbowl.auto.control.processor import generic_process_flow_logic
from punchbowl.auto.control.scheduler import generic_scheduler_flow_logic
from punchbowl.auto.flows.util import file_name_to_full_path, summarize_files_missing_cal_files
from punchbowl.level1.flow import level1_early_core_flow, level1_late_core_flow, level1_middle_core_flow

SCIENCE_LEVEL0_TYPE_CODES = ["PM", "PZ", "PP", "CR"]
SCIENCE_LEVEL1_MIDDLE_INPUT_TYPE_CODES = ["XM", "XZ", "XP"]
SCIENCE_LEVEL1_MIDDLE_OUTPUT_TYPE_CODES = ["YM", "YZ", "YP"]
SCIENCE_LEVEL1_LATE_INPUT_TYPE_CODES = ["YM", "YZ", "YP", "XR"]
SCIENCE_LEVEL1_LATE_INPUT_TYPE_CODES_NFI = ["XM", "XZ", "XP", "XR"]
SCIENCE_LEVEL1_LATE_OUTPUT_TYPE_CODES = ["PM", "PZ", "PP", "CR"]
SCIENCE_LEVEL1_QUICK_INPUT_TYPE_CODES = ["XR"]
SCIENCE_LEVEL1_QUICK_OUTPUT_TYPE_CODES = ["QR"]

@task(cache_policy=NO_CACHE)
def level1_early_query_ready_files(session, pipeline_config: dict, reference_time=None, max_n=9e99):
    logger = get_run_logger()
    ready = (session.query(File).filter(File.file_type.in_(SCIENCE_LEVEL0_TYPE_CODES))
                                .filter(File.state == "created")
                                .filter(File.level == "0"))

    target_date = pipeline_config.get("target_date")
    target_date = parse_datetime_str(target_date) if target_date else None
    dt = func.abs(func.timestampdiff(text("second"), File.date_obs, target_date)) if target_date else None
    if target_date:
        ready = ready.order_by(dt.asc())
    else:
        ready = ready.order_by(File.date_obs.desc())
    ready = ready.all()

    quartic_models = get_quartic_model_paths(ready, pipeline_config, session)
    vignetting_functions = get_vignetting_function_paths(ready, pipeline_config, session)
    mask_files = get_mask_files(ready, pipeline_config, session)
    actually_ready = []
    missing_quartic = []
    missing_vignetting = []
    missing_mask = []
    missing_sequence = []
    for f, quartic_model, vignetting_function, mask_file in zip(
            ready, quartic_models, vignetting_functions, mask_files):
        despike_neighbors = get_polarization_sequence(f, session=session)

        if quartic_model is None:
            missing_quartic.append(f)
            continue
        if vignetting_function[0] is None:
            missing_vignetting.append(f)
            continue
        if mask_file is None:
            missing_mask.append(f)
            continue
        if despike_neighbors is None or len(despike_neighbors) <= 2:
            missing_sequence.append(f)
        # Smuggle the identified models out of this function
        f.quartic_model = quartic_model
        f.vignetting_functions = vignetting_function
        f.mask_file = mask_file
        f.despike_neighbors = despike_neighbors
        actually_ready.append([f])
        if len(actually_ready) >= max_n:
            break
    if missing_quartic:
        logger.info("Missing quartic files for " + summarize_files_missing_cal_files(missing_quartic))
    if missing_vignetting:
        logger.info("Missing vignetting for " + summarize_files_missing_cal_files(missing_vignetting))
    if missing_mask:
        logger.info("Missing mask for " + summarize_files_missing_cal_files(missing_mask))
    if missing_sequence:
        logger.info("Missing despiking polarization sequence neighbors for "
                    + summarize_files_missing_cal_files(missing_sequence))
    return actually_ready

def get_polarization_sequence(f: File, session=None, crota_tolerance_degree=0.01, time_tolerance_minutes=15):
    neighbors = (session.query(File)
                 .filter(File.level == "0")
                 .filter(File.observatory == f.observatory)
                 .filter(or_(func.abs(File.crota - f.crota) < crota_tolerance_degree,
                             func.abs(File.crota - f.crota) > 360 - crota_tolerance_degree))
                 .filter(File.bad_packets == False)  # noqa: E712
                 .filter(File.date_obs != f.date_obs)  # do not include the image itself in the pol. sequence neighbors
                 .filter(File.date_obs > f.date_obs - timedelta(minutes=time_tolerance_minutes))
                 .filter(File.date_obs < f.date_obs + timedelta(minutes=time_tolerance_minutes)).all())
    return neighbors

def get_distortion_paths(level0_files, pipeline_config: dict, session=None):
    # Get all models, in reverse-chronological order
    models = (session.query(File)
              .filter(File.file_type == "DS")
              .where(File.file_version.not_like("v%")) #filters out "v0a"
              .order_by(File.file_version.desc(), File.date_obs.desc()).all())
    results = []
    for l0_file in level0_files:
        # We want to pick the latest model that's before the observation, so we go backwards in time, past any
        # later-in-time models, until we hit the first model that's before the observation.
        for model in models:
            if l0_file.observatory != model.observatory:
                continue
            if model.date_obs > l0_file.date_obs:
                continue
            results.append(model)
            break
        else:
            results.append(None)
    return results


def get_distortion_path(level0_file, pipeline_config: dict, session=None, reference_time=None):
    best_function = (session.query(File)
                     .filter(File.file_type == "DS")
                     .filter(File.observatory == level0_file.observatory)
                     .where(File.date_obs <= level0_file.date_obs)
                     .where(File.file_version.not_like("v%")) #filters out "v0a"
                     .order_by(File.file_version.desc(), File.date_obs.desc()).first())
    return best_function


VIGNETTING_CORRESPONDING_TYPES = {"PM": "GM",
                                  "PZ": "GZ",
                                  "PP": "GP",
                                  "CR": "GR"}


def get_vignetting_function_paths(level0_files, pipeline_config: dict, session=None):
    # Get all models, in reverse-chronological order
    models = (session.query(File)
              .filter(File.file_type.in_(["GM", "GZ", "GP", "GR"]))
              .where(File.file_version.not_like("v%")) #filters out "v0a".
              .order_by(File.file_version.desc(), File.date_obs.desc()).all())
    results = []
    for l0_file in level0_files:
        target_type = VIGNETTING_CORRESPONDING_TYPES[l0_file.file_type]
        # We want to pick the latest model that's before the observation, so we go backwards in time, past any
        # later-in-time models, until we hit the first model that's before the observation.
        before_model, after_model = None, None
        for model in models:
            if l0_file.observatory != model.observatory:
                continue
            if target_type != model.file_type:
                continue
            if model.date_obs > l0_file.date_obs:
                continue
            before_model = model
            break
        if l0_file.observatory == "4":
            for model in models[::-1]:
                if l0_file.observatory != model.observatory:
                    continue
                if target_type != model.file_type:
                    continue
                if model.date_obs < l0_file.date_obs:
                    continue
                after_model = model
                break
        results.append((before_model, after_model))
    return results


def get_vignetting_function_path(level0_file, pipeline_config: dict, session=None, reference_time=None):
    vignetting_function_type = VIGNETTING_CORRESPONDING_TYPES[level0_file.file_type]
    best_function = (session.query(File)
                     .filter(File.file_type == vignetting_function_type)
                     .filter(File.observatory == level0_file.observatory)
                     .where(File.date_obs <= level0_file.date_obs)
                     .where(File.file_version.not_like("v%")) #filters out "v0a".
                     .order_by(File.file_version.desc(), File.date_obs.desc())).first()
    if level0_file.observatory == "4":
        other_best_function = (session.query(File)
                               .filter(File.file_type == vignetting_function_type)
                               .filter(File.observatory == level0_file.observatory)
                               .where(File.date_obs >= level0_file.date_obs)
                               .where(File.file_version.not_like("v%"))  # filters out "v0a".
                               .order_by(File.file_version.desc(), File.date_obs.asc())).first()
        return best_function, other_best_function
    return best_function


PSF_MODEL_CORRESPONDING_TYPES = {"M": "RM",
                                 "Z": "RZ",
                                 "P": "RP",
                                 "R": "RC"}


def get_psf_model_paths(level0_files, pipeline_config: dict, session=None):
    # Get all models, in reverse-chronological order
    models = (session.query(File)
              .filter(File.file_type.startswith("R"))
              .where(File.file_version.not_like("v%")) #filters out "v0a".
              .order_by(File.file_version.desc(), File.date_obs.desc()).all())
    results = []
    for l0_file in level0_files:
        # TODO - Turn this back on once fine tuned for NFI
        if l0_file.observatory == "4":
            results.append("")
            continue
        target_type = PSF_MODEL_CORRESPONDING_TYPES[l0_file.file_type[1]]
        # We want to pick the latest model that's before the observation, so we go backwards in time, past any
        # later-in-time models, until we hit the first model that's before the observation.
        for model in models:
            if l0_file.observatory != model.observatory:
                continue
            if target_type != model.file_type:
                continue
            if model.date_obs > l0_file.date_obs:
                continue
            results.append(model.filename())
            break
        else:
            results.append(None)
    return results


def get_psf_model_path(level0_file, pipeline_config: dict, session=None, reference_time=None) -> str:
    psf_model_type = PSF_MODEL_CORRESPONDING_TYPES[level0_file.file_type[1]]
    # TODO - Turn this back on once fine tuned for NFI
    if level0_file.observatory == "4":
        return ""
    best_model = (session.query(File)
                  .filter(File.file_type == psf_model_type)
                  .filter(File.observatory == level0_file.observatory)
                  .where(File.date_obs <= level0_file.date_obs)
                  .where(File.file_version.not_like("v%")) #filters out "v0a".
                  .order_by(File.file_version.desc(), File.date_obs.desc()).first())
    return best_model.filename()

STRAY_LIGHT_CORRESPONDING_TYPES = {"M": "SM",
                                   "Z": "SZ",
                                   "P": "SP",
                                   "R": "SR"}

DYNAMIC_STRAY_LIGHT_CORRESPONDING_TYPES = {"M": "TM",
                                           "Z": "TZ",
                                           "P": "TP",
                                           "R": "TR"}


def get_two_closest_stray_light(X_files, session=None, max_distance: timedelta = None, dynamic=False):
    # Get all models
    models = (session.query(File)
              .filter(File.file_type.startswith("T" if dynamic else "S"))
              .filter(File.state == "created")
              .order_by(File.date_obs.asc()).all())
    models_by_pol_obs = defaultdict(list)
    for model in models:
        models_by_pol_obs[model.polarization + model.observatory].append(model)
    results = []
    for X_file in X_files:
        models = models_by_pol_obs[X_file.polarization + X_file.observatory]
        if max_distance:
            models = [m for m in models if abs(m.date_obs - X_file.date_obs) < max_distance]
        models = sorted(models, key=lambda m: abs(m.date_obs - X_file.date_obs))
        best_models = models[:2]
        if len(best_models) < 2:
            results.append((None, None))
        else:
            if best_models[1].date_obs < best_models[0].date_obs:
                best_models = best_models[::-1]
            results.append(best_models)
    return results


def get_two_best_stray_light(X_files, session=None, dynamic=False):
    # Get all models
    models = (session.query(File)
              .filter(File.file_type.startswith("T" if dynamic else "S"))
              .order_by(File.date_obs.asc()).all())
    models_by_pol_obs = defaultdict(list)
    for model in models:
        models_by_pol_obs[model.polarization + model.observatory].append(model)
    results = []
    for X_file in X_files:
        models = models_by_pol_obs[X_file.polarization + X_file.observatory]
        for before_model, after_model in pairwise(models):
            # All the models are sorted by date_obs, so there will be exactly one pair where the first is before our
            # file to be calibrated and the second is after
            if before_model.date_obs < X_file.date_obs < after_model.date_obs:
                break
        else:
            # We didn't find an appropriate pair, so we must still be waiting for the scheduler to fill in here and
            # tell us what's what
            results.append((None, None))
            continue

        if before_model.state == "created" and after_model.state == "created":
            # Good to go!
            results.append((before_model, after_model))
        elif before_model.state == "impossible" or after_model.state == "impossible":
            # Flexible mode---since we'll never be able to generate the "intended" models for this file, let's go for
            # the two closest possible models
            models = [m for m in models if m.state != "impossible"]
            models = sorted(models, key=lambda m: abs(m.date_obs - X_file.date_obs))
            before_model, after_model = models[:2]
            if after_model.date_obs < before_model.date_obs:
                before_model, after_model = after_model, before_model

            if before_model.state == "created" and after_model.state == "created":
                # Good to go!
                results.append((before_model, after_model))
            else:
                # Wait for files to generate
                results.append((None, None))
        else:
            # If we're here, we're waiting for at least one model to generate, but we do expect it to do so eventually
            results.append((None, None))
    return results


def get_first_last_stray_light(session, dynamic=False):
    target_type = "T%" if dynamic else "S%"
    dates = (session.query(func.min(File.date_obs), func.max(File.date_obs))
             .where(File.file_type.like(target_type)).
             where(File.state == "created")).all()
    if dates[0][0] is None:
        return datetime(1900, 1, 1), datetime(2900, 1, 1)
    return dates[0]


def get_quartic_model_paths(level0_files, pipeline_config: dict, session=None):
    # Get all models, in reverse-chronological order
    models = (session.query(File)
              .filter(File.file_type == "FQ")
              .where(File.file_version.not_like("v%")) #filters out "v0a".
              .order_by(File.file_version.desc(), File.date_obs.desc()).all())
    results = []
    for l0_file in level0_files:
        # We want to pick the latest model that's before the observation, so we go backwards in time, past any
        # later-in-time models, until we hit the first model that's before the observation.
        for model in models:
            if l0_file.observatory != model.observatory:
                continue
            if model.date_obs > l0_file.date_obs:
                continue
            results.append(model)
            break
        else:
            results.append(None)
    return results


def get_quartic_model_path(level0_file, pipeline_config: dict, session=None, reference_time=None):
    best_model = (session.query(File)
                  .filter(File.file_type == "FQ")
                  .filter(File.observatory == level0_file.observatory)
                  .where(File.date_obs <= level0_file.date_obs)
                  .where(File.file_version.not_like("v%")) #filters out "v0a".
                  .order_by(File.file_version.desc(), File.date_obs.desc()).first())
    return best_model


def get_mask_files(level0_files, pipeline_config: dict, session=None):
    # Get all models, in reverse-chronological order
    models = (session.query(File)
              .filter(File.file_type == "MS")
              .where(File.file_version.not_like("v%")) #filters out "v0a".
              .order_by(File.file_version.desc(), File.date_obs.desc()).all())
    results = []
    for l0_file in level0_files:
        # We want to pick the latest model that's before the observation, so we go backwards in time, past any
        # later-in-time models, until we hit the first model that's before the observation.
        for model in models:
            if l0_file.observatory != model.observatory:
                continue
            if model.date_obs > l0_file.date_obs:
                continue
            results.append(model)
            break
        else:
            results.append(None)
    return results


def get_mask_file(level0_file, pipeline_config: dict, session=None, reference_time=None):
    best_model = (session.query(File)
                  .filter(File.file_type == "MS")
                  .filter(File.observatory == level0_file.observatory)
                  .where(File.date_obs <= level0_file.date_obs)
                  .where(File.file_version.not_like("v%")) #filters out "v0a".
                  .order_by(File.file_version.desc(), File.date_obs.desc()).first())
    return best_model


def get_ccd_parameters(level0_file, pipeline_config: dict, session=None):
    gain_bottom, gain_top = pipeline_config["ccd_gain"][int(level0_file.observatory)]
    return {"gain_bottom": gain_bottom, "gain_top": gain_top}


def level1_early_construct_flow_info(level0_files: list[File], level1_files: list[File],
                               pipeline_config: dict, session=None, reference_time=None):
    flow_type = "level1_early"
    state = "planned"
    creation_time = datetime.now()
    priority = pipeline_config["flows"][flow_type]["priority"]["initial"]

    before_vignetting_function = level0_files[0].vignetting_functions[0]
    after_vignetting_function = level0_files[0].vignetting_functions[1]
    if after_vignetting_function is not None:
        after_vignetting_function = after_vignetting_function.filename()
    best_quartic_model = level0_files[0].quartic_model
    despike_neighbors = level0_files[0].despike_neighbors
    ccd_parameters = get_ccd_parameters(level0_files[0], pipeline_config, session=session)
    mask_function = level0_files[0].mask_file

    call_data = json.dumps(
        {
            "input_data": [level0_file.filename() for level0_file in level0_files],
            "vignetting_function_path": before_vignetting_function.filename(),
            "second_vignetting_function_path": after_vignetting_function,
            "quartic_coefficient_path": best_quartic_model.filename(),
            "gain_bottom": ccd_parameters["gain_bottom"],
            "gain_top": ccd_parameters["gain_top"],
            "despike_neighbors": [n.filename() for n in despike_neighbors],
            "mask_path": mask_function.filename().replace(".fits", ".bin"),
        },
    )
    return Flow(
        flow_type=flow_type,
        flow_level="1",
        state=state,
        creation_time=creation_time,
        priority=priority,
        call_data=call_data,
    )


def level1_early_construct_file_info(level0_files: list[File], pipeline_config: dict, reference_time=None) -> list[File]:
    files = []
    files.append(File(
            level="1",
            file_type="X" + level0_files[0].file_type[1:],
            observatory=level0_files[0].observatory,
            file_version=pipeline_config["file_version"],
            software_version=__version__,
            date_obs=level0_files[0].date_obs,
            polarization=level0_files[0].polarization,
            outlier=level0_files[0].outlier,
            bad_packets=level0_files[0].bad_packets,
            state="planned",
        ))
    return files


@flow
def level1_early_scheduler_flow(pipeline_config_path=None, session=None, reference_time=None):
    generic_scheduler_flow_logic(
        level1_early_query_ready_files,
        level1_early_construct_file_info,
        level1_early_construct_flow_info,
        pipeline_config_path,
        reference_time=reference_time,
        session=session,
    )


def level1_early_call_data_processor(call_data: dict, pipeline_config, session=None) -> dict:
    for key in ["input_data", "quartic_coefficient_path", "vignetting_function_path",
                "second_vignetting_function_path", "mask_path"]:
        call_data[key] = file_name_to_full_path(call_data[key], pipeline_config["root"])

    call_data["quartic_coefficient_path"] = cache_layer.quartic_coefficients.wrap_if_appropriate(
        call_data["quartic_coefficient_path"])
    call_data["vignetting_function_path"] = cache_layer.vignetting_function.wrap_if_appropriate(
        call_data["vignetting_function_path"])
    if call_data["second_vignetting_function_path"] is not None:
        call_data["second_vignetting_function_path"] = cache_layer.vignetting_function.wrap_if_appropriate(
            call_data["second_vignetting_function_path"])

    call_data["despike_neighbors"] = [file_name_to_full_path(p, pipeline_config["root"])
                                      for p in call_data["despike_neighbors"]]

    # Anything more than 16 doesn't offer any real benefit, and the default of n_cpu on punch190 is actually slower than
    # 16! Here we choose less to have less spiky CPU usage to play better with other flows.
    call_data["max_workers"] = 2
    return call_data


@flow
def level1_early_process_flow(flow_id: int | list[int], pipeline_config_path=None, session=None):
    generic_process_flow_logic(flow_id, level1_early_core_flow, pipeline_config_path, session=session,
                               call_data_processor=level1_early_call_data_processor)


@task(cache_policy=NO_CACHE)
def level1_middle_query_ready_files(session, pipeline_config: dict, reference_time=None, max_n=9e99):
    logger = get_run_logger()
    start_date, end_date = get_first_last_stray_light(session, dynamic=True)
    child = aliased(File)
    child_exists_subquery = (session.query(FileRelationship)
                             .join(child, FileRelationship.child == child.file_id)
                             .filter(FileRelationship.parent == File.file_id)
                             .filter(child.file_type.in_(SCIENCE_LEVEL1_MIDDLE_OUTPUT_TYPE_CODES))
                             .exists())
    ready = (session.query(File)
             .filter(File.file_type.in_(SCIENCE_LEVEL1_MIDDLE_INPUT_TYPE_CODES))
             .filter(File.level == "1")
             .filter(File.state.in_(["created", "progressed"]))
             .filter(~child_exists_subquery)
             .filter(File.date_obs >= start_date)
             .filter(File.date_obs <= end_date))

    target_date = pipeline_config.get("target_date")
    target_date = parse_datetime_str(target_date) if target_date else None
    dt = func.abs(func.timestampdiff(text("second"), File.date_obs, target_date)) if target_date else None
    if target_date:
        ready = ready.order_by(dt.asc())
    else:
        ready = ready.order_by(File.date_obs.desc())
    ready = ready.all()

    actually_ready = []
    missing_stray_light = []

    best_stray_lights = get_two_best_stray_light(ready, session=session, dynamic=True)

    for f, best_stray_light in zip(ready, best_stray_lights):
        if best_stray_light == (None, None):
            missing_stray_light.append(f)
            continue
        f.dynamic_stray_light = best_stray_light
        actually_ready.append([f])
        if len(actually_ready) >= max_n:
            break
    if missing_stray_light:
        logger.info("Waiting for dynamic stray light models for " + summarize_files_missing_cal_files(missing_stray_light))
    return actually_ready


def level1_middle_construct_flow_info(input_files: list[File], output_files: list[File],
                                    pipeline_config: dict, session=None, reference_time=None):
    flow_type = "level1_middle"
    state = "planned"
    creation_time = datetime.now()
    priority = pipeline_config["flows"][flow_type]["priority"]["initial"]

    dynamic_stray_light_before, dynamic_stray_light_after = input_files[0].dynamic_stray_light

    call_data = json.dumps(
        {
            "input_data": [input_file.filename() for input_file in input_files],
            "dynamic_stray_light_before_path":
                dynamic_stray_light_before.filename() if dynamic_stray_light_before else None,
            "dynamic_stray_light_after_path":
                dynamic_stray_light_after.filename() if dynamic_stray_light_after else None,
        },
    )
    return Flow(
        flow_type=flow_type,
        flow_level="1",
        state=state,
        creation_time=creation_time,
        priority=priority,
        call_data=call_data,
    )


def level1_middle_construct_file_info(input_files: list[File], pipeline_config: dict, reference_time=None) -> list[File]:
    return [
        File(
            level="1",
            file_type="Y" + input_files[0].file_type[1:],
            observatory=input_files[0].observatory,
            file_version=pipeline_config["file_version"],
            software_version=__version__,
            date_obs=input_files[0].date_obs,
            polarization=input_files[0].polarization,
            outlier=input_files[0].outlier,
            bad_packets=input_files[0].bad_packets,
            state="planned",
        ),
    ]


@flow
def level1_middle_scheduler_flow(pipeline_config_path=None, session=None, reference_time=None):
    generic_scheduler_flow_logic(
        level1_middle_query_ready_files,
        level1_middle_construct_file_info,
        level1_middle_construct_flow_info,
        pipeline_config_path,
        reference_time=reference_time,
        session=session,
    )


def level1_middle_call_data_processor(call_data: dict, pipeline_config, session=None) -> dict:
    for key in ["input_data", "dynamic_stray_light_before_path", "dynamic_stray_light_after_path"]:
        call_data[key] = file_name_to_full_path(call_data[key], pipeline_config["root"])

    call_data["dynamic_stray_light_before_path"] = cache_layer.stray_light.wrap_if_appropriate(
            call_data["dynamic_stray_light_before_path"])
    call_data["dynamic_stray_light_after_path"] = cache_layer.stray_light.wrap_if_appropriate(
            call_data["dynamic_stray_light_after_path"])
    return call_data


@flow
def level1_middle_process_flow(flow_id: int | list[int], pipeline_config_path=None, session=None):
    generic_process_flow_logic(flow_id, level1_middle_core_flow, pipeline_config_path, session=session,
                               call_data_processor=level1_middle_call_data_processor)


@task(cache_policy=NO_CACHE)
def level1_late_query_ready_files(session, pipeline_config: dict, reference_time=None, max_n=9e99):
    logger = get_run_logger()
    start_date, end_date = get_first_last_stray_light(session)
    child = aliased(File)
    child_exists_subquery = (session.query(FileRelationship)
                             .join(child, FileRelationship.child == child.file_id)
                             .filter(FileRelationship.parent == File.file_id)
                             .filter(child.file_type.in_(SCIENCE_LEVEL1_LATE_OUTPUT_TYPE_CODES))
                             .exists())
    ready = (session.query(File)
             .filter(or_(
                and_(File.file_type.in_(SCIENCE_LEVEL1_LATE_INPUT_TYPE_CODES),
                     File.observatory.in_(['1', '2', '3'])
                     ),
                and_(File.file_type.in_(SCIENCE_LEVEL1_LATE_INPUT_TYPE_CODES_NFI),
                     File.observatory == '4'
                     ),
             ))
             .filter(File.level == "1")
             .filter(File.state.in_(["created", "progressed"]))
             .filter(~child_exists_subquery)
             .filter(File.date_obs >= start_date)
             .filter(File.date_obs <= end_date))

    target_date = pipeline_config.get("target_date")
    target_date = parse_datetime_str(target_date) if target_date else None
    dt = func.abs(func.timestampdiff(text("second"), File.date_obs, target_date)) if target_date else None
    if target_date:
        ready = ready.order_by(dt.asc())
    else:
        ready = ready.order_by(File.date_obs.desc())
    ready = ready.all()


    distortion_paths = get_distortion_paths(ready, pipeline_config, session)
    psf_paths = get_psf_model_paths(ready, pipeline_config, session)
    best_stray_lights = get_two_best_stray_light(ready, session=session, dynamic=False)
    actually_ready = []
    missing_stray_light = []
    missing_distortion = []
    missing_psf = []

    for f, distortion_path, psf_path, best_stray_light in zip(ready, distortion_paths, psf_paths, best_stray_lights):
        if best_stray_light == (None, None):
            missing_stray_light.append(f)
            continue
        if distortion_path is None:
            missing_distortion.append(f)
            continue
        if psf_path is None:
            missing_psf.append(f)
            continue
        f.distortion_path = distortion_path
        f.psf_path = psf_path
        f.stray_light = best_stray_light
        actually_ready.append([f])
        if len(actually_ready) >= max_n:
            break
    if missing_stray_light:
        logger.info("Waiting for stray light models for " + summarize_files_missing_cal_files(missing_stray_light))
    if missing_distortion:
        logger.info("Missing distortion for " + summarize_files_missing_cal_files(missing_distortion))
    if missing_psf:
        logger.info("Missing PSF for " + summarize_files_missing_cal_files(missing_psf))
    # It's easiest to batch-query here, where we have all the File objects in one list
    masks = get_mask_files([f[0] for f in actually_ready], pipeline_config, session)
    for f, mask in zip(actually_ready, masks):
        f[0].mask_path = mask
    return actually_ready


def level1_late_construct_flow_info(input_files: list[File], output_files: list[File],
                                    pipeline_config: dict, session=None, reference_time=None):
    flow_type = "level1_late"
    state = "planned"
    creation_time = datetime.now()
    priority = pipeline_config["flows"][flow_type]["priority"]["initial"]

    best_psf_model = input_files[0].psf_path
    best_distortion = input_files[0].distortion_path
    stray_light_before, stray_light_after = input_files[0].stray_light
    mask_function = input_files[0].mask_path

    call_data = json.dumps(
        {
            "input_data": [input_file.filename() for input_file in input_files],
            "psf_model_path": best_psf_model,
            "distortion_path": best_distortion.filename(),
            "stray_light_before_path": stray_light_before.filename() if stray_light_before else None,
            "stray_light_after_path": stray_light_after.filename() if stray_light_after else None,
            "mask_path": mask_function.filename().replace(".fits", ".bin"),
            "output_as_Q_file": False,
        },
    )
    return Flow(
        flow_type=flow_type,
        flow_level="1",
        state=state,
        creation_time=creation_time,
        priority=priority,
        call_data=call_data,
    )


def level1_late_construct_file_info(input_files: list[File], pipeline_config: dict, reference_time=None) -> list[File]:
    prefix = "C" if input_files[0].polarization == "C" else "P"
    return [
        File(
            level="1",
            file_type=prefix + input_files[0].file_type[1:],
            observatory=input_files[0].observatory,
            file_version=pipeline_config["file_version"],
            software_version=__version__,
            date_obs=input_files[0].date_obs,
            polarization=input_files[0].polarization,
            outlier=input_files[0].outlier,
            bad_packets=input_files[0].bad_packets,
            state="planned",
        ),
    ]


@flow
def level1_late_scheduler_flow(pipeline_config_path=None, session=None, reference_time=None):
    generic_scheduler_flow_logic(
        level1_late_query_ready_files,
        level1_late_construct_file_info,
        level1_late_construct_flow_info,
        pipeline_config_path,
        reference_time=reference_time,
        session=session,
    )


def level1_late_call_data_processor(call_data: dict, pipeline_config, session=None) -> dict:
    for key in ["input_data", "mask_path", "stray_light_before_path", "stray_light_after_path", "distortion_path"]:
        call_data[key] = file_name_to_full_path(call_data[key], pipeline_config["root"])

    # TODO: this is a hack to skip NFI PSF. Remove!
    if call_data["psf_model_path"] == "":
        call_data["psf_model_path"] = None
    else:
        call_data["psf_model_path"] = file_name_to_full_path(call_data["psf_model_path"], pipeline_config["root"])
        call_data["psf_model_path"] = cache_layer.psf.wrap_if_appropriate(call_data["psf_model_path"])

    # Anything more than 16 doesn't offer any real benefit, and the default of n_cpu on punch190 is actually slower than
    # 16! Here we choose less to have less spiky CPU usage to play better with other flows.
    call_data["max_workers"] = 2

    call_data["stray_light_before_path"] = cache_layer.stray_light.wrap_if_appropriate(
            call_data["stray_light_before_path"])
    call_data["stray_light_after_path"] = cache_layer.stray_light.wrap_if_appropriate(
            call_data["stray_light_after_path"])
    return call_data


@flow
def level1_late_process_flow(flow_id: int | list[int], pipeline_config_path=None, session=None):
    generic_process_flow_logic(flow_id, level1_late_core_flow, pipeline_config_path, session=session,
                               call_data_processor=level1_late_call_data_processor)


@task(cache_policy=NO_CACHE)
def level1_quick_query_ready_files(session, pipeline_config: dict, reference_time=None, max_n=9e99):
    logger = get_run_logger()
    child = aliased(File)
    no_earlier_than = pipeline_config["flows"]["level1_quick"].get("no-earlier-than", "1970-01-01")
    child_exists_subquery = (session.query(FileRelationship)
                             .join(child, FileRelationship.child == child.file_id)
                             .filter(FileRelationship.parent == File.file_id)
                             .filter(child.file_type.in_(SCIENCE_LEVEL1_QUICK_OUTPUT_TYPE_CODES))
                             .exists())
    ready = (session.query(File)
             .filter(File.file_type.in_(SCIENCE_LEVEL1_QUICK_INPUT_TYPE_CODES))
             .filter(File.level == "1")
             .filter(File.state.in_(["created", "progressed"]))
             .filter(File.date_obs >= no_earlier_than)
             .filter(~child_exists_subquery)
             .order_by(File.date_obs.desc()).all())

    actually_ready = []
    missing_stray_light = []
    missing_distortion = []
    missing_psf = []
    distortion_paths = get_distortion_paths(ready, pipeline_config, session)
    psf_paths = get_psf_model_paths(ready, pipeline_config, session)
    stray_lights = get_two_closest_stray_light(ready, session=session, dynamic=False)
    for f, distortion_path, psf_path, closest_stray_light in zip(ready, distortion_paths, psf_paths, stray_lights):
        if closest_stray_light == [None, None]:
            missing_stray_light.append(f)
            continue
        if distortion_path is None:
            missing_distortion.append(f)
            continue
        if psf_path is None:
            missing_psf.append(f)
            continue
        f.distortion_path = distortion_path
        f.psf_path = psf_path
        f.stray_light = closest_stray_light
        actually_ready.append([f])
        if len(actually_ready) >= max_n:
            break
    if missing_stray_light:
        logger.info("Waiting for stray light models for " + summarize_files_missing_cal_files(missing_stray_light))
    if missing_distortion:
        logger.info("Missing distortion for " + summarize_files_missing_cal_files(missing_distortion))
    if missing_psf:
        logger.info("Missing PSF for " + summarize_files_missing_cal_files(missing_psf))
    # It's easiest to batch-query here, where we have all the File objects in one list
    masks = get_mask_files([f[0] for f in actually_ready], pipeline_config, session)
    for f, mask in zip(actually_ready, masks):
        f[0].mask_path = mask
    return actually_ready


def level1_quick_construct_flow_info(input_files: list[File], output_files: list[File],
                                    pipeline_config: dict, session=None, reference_time=None):
    flow_type = "level1_quick"
    state = "planned"
    creation_time = datetime.now()
    priority = pipeline_config["flows"][flow_type]["priority"]["initial"]

    best_psf_model = input_files[0].psf_path
    best_distortion = input_files[0].distortion_path
    stray_light_before, stray_light_after = input_files[0].stray_light
    mask_function = input_files[0].mask_path

    call_data = json.dumps(
        {
            "input_data": [input_file.filename() for input_file in input_files],
            "psf_model_path": best_psf_model,
            "distortion_path": best_distortion.filename(),
            "stray_light_before_path": stray_light_before.filename() if stray_light_before else None,
            "stray_light_after_path": stray_light_after.filename() if stray_light_after else None,
            "mask_path": mask_function.filename().replace(".fits", ".bin"),
            "output_as_Q_file": True,
        },
    )
    return Flow(
        flow_type=flow_type,
        flow_level="1",
        state=state,
        creation_time=creation_time,
        priority=priority,
        call_data=call_data,
    )


def level1_quick_construct_file_info(input_files: list[File], pipeline_config: dict, reference_time=None) -> list[File]:
    return [
        File(
            level="1",
            file_type="Q" + input_files[0].file_type[1:],
            observatory=input_files[0].observatory,
            file_version=pipeline_config["file_version"],
            software_version=__version__,
            date_obs=input_files[0].date_obs,
            polarization=input_files[0].polarization,
            outlier=input_files[0].outlier,
            bad_packets=input_files[0].bad_packets,
            state="planned",
        ),
    ]


@flow
def level1_quick_scheduler_flow(pipeline_config_path=None, session=None, reference_time=None):
    generic_scheduler_flow_logic(
        level1_quick_query_ready_files,
        level1_quick_construct_file_info,
        level1_quick_construct_flow_info,
        pipeline_config_path,
        reference_time=reference_time,
        session=session,
    )


def level1_quick_call_data_processor(call_data: dict, pipeline_config, session=None) -> dict:
    for key in ["input_data", "mask_path", "stray_light_before_path", "stray_light_after_path", "distortion_path"]:
        call_data[key] = file_name_to_full_path(call_data[key], pipeline_config["root"])

    # TODO: this is a hack to skip NFI PSF. Remove!
    if call_data["psf_model_path"] == "":
        call_data["psf_model_path"] = None
    else:
        call_data["psf_model_path"] = file_name_to_full_path(call_data["psf_model_path"], pipeline_config["root"])
        call_data["psf_model_path"] = cache_layer.psf.wrap_if_appropriate(call_data["psf_model_path"])

    # Anything more than 16 doesn't offer any real benefit, and the default of n_cpu on punch190 is actually slower than
    # 16! Here we choose less to have less spiky CPU usage to play better with other flows.
    call_data["max_workers"] = 2
    return call_data


@flow
def level1_quick_process_flow(flow_id: int | list[int], pipeline_config_path=None, session=None):
    generic_process_flow_logic(flow_id, level1_late_core_flow, pipeline_config_path, session=session,
                               call_data_processor=level1_quick_call_data_processor)
