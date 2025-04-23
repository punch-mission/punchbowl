from typing import Any

from ndcube import NDCube
from prefect import Flow, Task, flow, get_run_logger, task
from prefect.client.schemas.objects import TaskRun
from prefect.states import State
from prefect.variables import Variable

from punchbowl.data.punch_io import get_base_file_name, write_ndcube_to_fits


def completion_debugger(task: Task, task_run: TaskRun, state: State) -> None:
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
            logger = get_run_logger()
            logger.error(f"Cannot write debug output for {task} {task_run} in {state}.")

def failure_hook(task: Task, task_run: TaskRun, state: State) -> None:
    """Run if a punch_task fails."""

def punch_task(*args: Any, **kwargs: Any) -> Task:
    """Prefect task that does PUNCH special things."""
    return task(*args, **kwargs, on_completion=[completion_debugger], on_failure=[failure_hook])

def punch_flow(*args: Any, **kwargs: Any) -> Flow:
    """Prefect flow that does PUNCH special things."""
    return flow(*args, **kwargs, validate_parameters=False)
