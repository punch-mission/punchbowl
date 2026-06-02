import logging
from typing import Any
from functools import cache
from collections.abc import Callable

from httpx import ConnectError
from prefect import Flow, Task, flow, get_run_logger, runtime, task
from prefect.cache_policies import NO_CACHE
from prefect.client.schemas.objects import TaskRun
from prefect.states import State
from prefect.variables import Variable

from punchbowl.data.punch_io import get_base_file_name, write_ndcube_to_fits
from punchbowl.data.punchcube import PUNCHCube


def completion_debugger(task: Task, task_run: TaskRun, state: State) -> None:
    """Run on task completion during debug mode."""
    if Variable.get("debug", False):
        cube = state.result()
        if isinstance(cube, PUNCHCube):
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

try:
    _debug_mode = Variable.get("debug", False)
except (ConnectError, RuntimeError):
    _debug_mode = False


def punch_task(func: Callable | None = None, **kwargs: Any) -> Task | Callable:
    """Prefect task that does PUNCH special things."""
    if detect_if_running_in_prefect():
        # Delegate everything to Prefect
        return task(func, **kwargs,
                    on_completion=[completion_debugger] if _debug_mode else [],
                    on_failure=[failure_hook],
                    cache_policy=NO_CACHE)
    if func is None:
        # We've been used as @punch_task() or @punch_task(arg=val), so we are to return a function that does the
        # decoration
        return _compatability_decorator
    # We've been used as @punch_task, so we are to do the decoration directly
    return _compatability_decorator(func)


def punch_flow(func: Callable | None = None, **kwargs: Any) -> Flow | Callable:
    """Prefect flow that does PUNCH special things."""
    if detect_if_running_in_prefect():
        # Delegate everything to Prefect
        return flow(func, **kwargs, validate_parameters=False)
    if func is None:
        # We've been used as @punch_task() or @punch_task(arg=val), so we are to return a function that does the
        # decoration
        return _compatability_decorator
    # We've been used as @punch_task, so we are to do the decoration directly
    return _compatability_decorator(func)


def _compatability_decorator(func: Callable) -> Callable:
    """Make wrapped functions have a .fn attribute like Prefect Flows and Tasks."""
    func.fn = func
    return func


@cache
def detect_if_running_in_prefect() -> bool:
    """Determine if we're running under Prefect."""
    return runtime.flow_run.name is not None


def get_logger() -> logging.Logger:
    """Get a logger, which will be the Prefect logger if we're running under Prefect."""
    if detect_if_running_in_prefect():
        return get_run_logger()
    return logging.getLogger("punchbowl")
