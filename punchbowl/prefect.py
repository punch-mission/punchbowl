# ruff: noqa
import os

from prefect import task
from prefect.variables import Variable

from punchbowl.data.io import get_base_file_name, write_ndcube_to_fits


def completion_debugger(task, task_run, state):
    """Run on task completion during debug mode."""
    if Variable.get("debug", False):
        new_filename = f"{get_base_file_name(state)}_{task.name}.fits"
        write_ndcube_to_fits(state, new_filename, overwrite=True)

def failure_hook(task, task_run, state):
    """Run if a punch_task fails."""
    print(f"I'm a special debugger that runs if {task.name} fails. I failed on {task_run} with {state}."
          "I could also be configured to alert the SOC I failed.")

def punch_task(*args, **kwargs):
    """Prefect task that does PUNCH special things."""
    return task(*args, **kwargs, on_completion=[completion_debugger], on_failure=[failure_hook])
