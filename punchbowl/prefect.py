# ruff: noqa
import os

from ndcube import NDCube
from prefect import task
from prefect.variables import Variable

from punchbowl.data.io import get_base_file_name, write_ndcube_to_fits


def completion_debugger(task, task_run, state):
    """Run on task completion during debug mode."""
    if Variable.get("debug", False):
        cube = state.result()
        if isinstance(cube, NDCube):
            new_filename = f"{get_base_file_name(cube)}_{task.name}.fits"
            write_ndcube_to_fits(cube, new_filename, overwrite=True, write_hash=False)
        elif isinstance(cube, list):
            for i, c in enumerate(cube):
                new_filename = f"{get_base_file_name(c)}_{task.name}_{i}.fits"
                write_ndcube_to_fits(c, new_filename, overwrite=True, write_hash=False)
        else:
            print("Cannot write output.")

def failure_hook(task, task_run, state):
    """Run if a punch_task fails."""
    print(f"I'm a special debugger that runs if {task.name} fails. I failed on {task_run} with {state}."
          "I could also be configured to alert the SOC I failed.")

def punch_task(*args, **kwargs):
    """Prefect task that does PUNCH special things."""
    return task(*args, **kwargs, on_completion=[completion_debugger], on_failure=[failure_hook])
