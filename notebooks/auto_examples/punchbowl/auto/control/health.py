from datetime import datetime

import psutil
from prefect import flow

from punchbowl.auto.control.db import Health
from punchbowl.auto.control.util import get_database_session, load_pipeline_configuration


@flow
def health_monitor(pipeline_config_path: str = None):
    config = load_pipeline_configuration(pipeline_config_path)

    now = datetime.now()
    cpu_usage = psutil.cpu_percent(interval=5)
    memory_usage = psutil.virtual_memory().used / 1E9  # store in GB
    memory_percentage = psutil.virtual_memory().percent
    disk_usage = psutil.disk_usage(config.get("root", "/")).used / 1E9  # store in GB
    disk_percentage = psutil.disk_usage(config.get("root", "/")).percent
    num_pids = len(psutil.pids())

    with get_database_session() as session:
        new_health_entry = Health(datetime=now,
                                  cpu_usage=cpu_usage,
                                  memory_usage=memory_usage,
                                  memory_percentage=memory_percentage,
                                  disk_usage=disk_usage,
                                  disk_percentage=disk_percentage,
                                  num_pids=num_pids)
        session.add(new_health_entry)
        session.commit()


# TODO add zombie monitor!
