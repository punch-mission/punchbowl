import os
import json
from datetime import UTC, datetime

from dateutil.parser import parse as parse_datetime_str
from prefect import get_run_logger, tags
from prefect.context import MissingContextError, get_run_context

from punchbowl.auto.control.db import File, Flow
from punchbowl.auto.control.util import (
    get_database_session,
    load_pipeline_configuration,
    match_data_with_file_db_entry,
    write_file,
)


def generic_process_flow_logic(flow_id: int | list[int], core_flow_to_launch, pipeline_config_path: str, session=None,
                               call_data_processor=None ):
    if session is None:
        session = get_database_session()
    if isinstance(flow_id, int):
        flow_ids = [flow_id]
    else:
        flow_ids = flow_id

    try:
        logger = get_run_logger()
        logger.info(f"Running under PID {os.getpid()}")

        # load pipeline configuration
        pipeline_config = load_pipeline_configuration(pipeline_config_path)

        flow_db_entries = session.query(Flow).where(Flow.flow_id.in_(flow_ids)).all()
        if len(flow_db_entries) != len(flow_ids):
            raise RuntimeError("Did not find the right number of flows")

        file_db_entry_lists = []

        for flow_db_entry in flow_db_entries:
            if flow_db_entry.state != "launched":
                raise RuntimeError(f"Flow id {flow_db_entry.flow_id} has state '{flow_db_entry.state}'; not running")
            logger.info(f"Will be running on flow db entry with id={flow_db_entry.flow_id}, scheduled at "
                        f"{flow_db_entry.creation_time} and launched at {flow_db_entry.launch_time}.")

            # update the processing flow name with the flow run name from Prefect
            try:
                flow_run_context = get_run_context()
                flow_db_entry.flow_run_name = flow_run_context.flow_run.name
                flow_db_entry.flow_run_id = flow_run_context.flow_run.id
            except MissingContextError:
                # We're not in a flow context---probably we're running under speedster
                pass
            flow_db_entry.state = "running"
            flow_db_entry.start_time = datetime.now()

            file_db_entry_list = session.query(File).where(File.processing_flow == flow_db_entry.flow_id).all()

            # update the file database entries as being created
            try:
                if file_db_entry_list:
                    for file_db_entry in file_db_entry_list:
                        if file_db_entry.state != "planned":
                            raise RuntimeError(f"File id {file_db_entry.file_id} has already been created.")
                        if os.path.exists(os.path.join(
                                file_db_entry.directory(pipeline_config["root"]), file_db_entry.filename())):
                            raise RuntimeError(f"Expected output file {file_db_entry.filename()} (id {file_db_entry.file_id}) "
                                                "already exists on disk")
                        file_db_entry.state = "creating"
                else:
                    raise RuntimeError("There should be at least one file associated with this flow. Found 0.")
            except:
                # The exception handler rolls back the transaction, but we do want our start_time to stay in place. So
                # commit on error, but otherwise let the transaction keep growing into a big batch
                session.commit()
                raise
            file_db_entry_lists.append(file_db_entry_list)
        session.commit()

        for flow_db_entry, file_db_entry_list in zip(flow_db_entries, file_db_entry_lists):
            # load the call data and launch the core flow
            flow_call_data = json.loads(flow_db_entry.call_data)
            if call_data_processor is not None:
                flow_call_data = call_data_processor(flow_call_data, pipeline_config, session)
            output_file_ids = set()
            expected_file_ids = {entry.file_id for entry in file_db_entry_list}
            logger.info(f"Expecting to output files with ids={expected_file_ids}.")

            tag_set = {entry.file_type + entry.observatory for entry in file_db_entry_list}
            with tags(*sorted(tag_set)):
                results = core_flow_to_launch(**flow_call_data)
            for result in results:
                result.meta["FILEVRSN"] = pipeline_config["file_version"]
                file_db_entry = match_data_with_file_db_entry(result, file_db_entry_list)
                logger.info(f"Preparing to write {file_db_entry.file_id}.")
                output_file_ids.add(file_db_entry.file_id)
                file_db_entry.date_obs = parse_datetime_str(result.meta["DATE-OBS"].value)
                file_db_entry.state = "created"
                date_created = parse_datetime_str(result.meta["DATE"].value)
                # Converts to local time
                date_created = date_created.replace(tzinfo=UTC).astimezone()
                file_db_entry.date_created = date_created
                result.meta["OUTLIER"] = int(file_db_entry.outlier)
                result.meta["BADPKTS"] = int(file_db_entry.bad_packets)
                filename = write_file(result, file_db_entry, pipeline_config)
                logger.info(f"Wrote to {filename}")

            missing_file_ids = expected_file_ids.difference(output_file_ids)
            if missing_file_ids:
                raise RuntimeError(f"We did not get an output cube for file ids {missing_file_ids}")

        session.commit()

        for flow_db_entry, file_db_entry_list in zip(flow_db_entries, file_db_entry_lists):
            # Don't overwrite if our flow state has been changed from under us (e.g. it's been changed to 'revivable')
            session.refresh(flow_db_entry)
            if flow_db_entry.state == "running":
                flow_db_entry.state = "completed"
            flow_db_entry.end_time = datetime.now()
            # Note: the file_db_entry gets updated above in the writing step because it could be created or blank
        session.commit()
    except:
        session.rollback()
        session.query(Flow).filter(Flow.flow_id.in_(flow_ids)).update(
            {"state": "failed",
             "end_time": datetime.now()})
        session.query(File).filter(File.processing_flow.in_(flow_ids)).update(
            {"state": "failed"})
        session.commit()
        raise
