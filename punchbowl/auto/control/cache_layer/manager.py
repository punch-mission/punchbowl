import contextlib
from glob import glob
from multiprocessing.shared_memory import SharedMemory

from prefect import get_run_logger
from prefect.exceptions import MissingContextError
from prefect.variables import Variable

CACHE_KEY_PREFIX = "punchpipe-cache-"

def caching_is_enabled() -> bool:
    return Variable.get("use_shm_cache", False)


class ExportableWrapper:
    """
    Allows shared memory buffers to be yielded without "leaking" a reference

    If try_read_from_key just does `yield shm.buf`, then the calling function will still have a reference to our
    shared memory buffer when we enter our `finally` block and try to close the memory. By instead yielding this
    wrapper, we can invalidate its reference to the shared memory and close it.
    """

    def __init__(self, buffer):
        self.data = buffer


@contextlib.contextmanager
def try_read_from_key(key) -> ExportableWrapper | None:
    shm = None
    try:
        shm = SharedMemory(CACHE_KEY_PREFIX + key, track=False)
        if shm.buf[0] == 1:
            wrapper = ExportableWrapper(shm.buf[1:])
            yield wrapper
            wrapper.data = None
        else:
            yield None
    except FileNotFoundError:
        yield None
    finally:
        if shm is not None:
            shm.close()


def try_write_to_key(key, data):
    shm = None
    try:
        shm = SharedMemory(CACHE_KEY_PREFIX + key, create=True, size=len(data) + 1, track=False)
        # buf[0] will be a sentinel value to indicate that the rest of the data is in place, in case another process
        # opens this shared memory while we're still copying
        shm.buf[0] = 0
        shm.buf[1:len(data)+1] = data
        shm.buf[0] = 1
        try:
            get_run_logger().info(f"Saved to cache key {key}")
        except MissingContextError:
            pass  # we're not in a flow so we don't log
    except FileExistsError:
        pass
    finally:
        if shm is not None:
            shm.close()


def get_existing_cache_files():
    return glob(f"/dev/shm/{CACHE_KEY_PREFIX}*")
