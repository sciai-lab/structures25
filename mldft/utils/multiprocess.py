import multiprocessing
import os
from functools import wraps

import psutil
import torch
from loguru import logger
from pyscf import lib


def set_num_threads(num_threads: int):
    """Set the number of threads to use in pyscf and numpy and torch."""
    # Locally pyscf lib works, on the cluster os.environ["OMP_NUM_THREADS"] = str(num_threads_per_process) is needed
    lib.num_threads(num_threads)
    # Other numpy backends might need other environment variables to be set, but this has worked for our purposes
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)


def configure_processes_and_threads(
    num_processes: int | None = None,
    num_threads_per_process: int | None = None,
) -> tuple[int, int]:
    """Configure the number of processes and threads per process and set them in pyscf and numpy.

    If the number of threads is not specified, it is
    defaulted to 1. If the number of processes is not specified, the maximum available number of processes given the
    number of cpu cores and threads per process is used.

    Args:
        num_processes: The number of processes to use.
        num_threads_per_process: The number of threads per process to use.
    """
    num_threads_per_process = num_threads_per_process if num_threads_per_process is not None else 1
    num_processes = (
        num_processes
        if num_processes is not None
        else max((multiprocessing.cpu_count() - 1) // num_threads_per_process, 1)
    )
    set_num_threads(num_threads_per_process)
    return num_processes, num_threads_per_process


def get_memory_usage_in_mb():
    """Get the memory usage of the current process in MB."""
    process = psutil.Process()
    memory_info = process.memory_info()
    mega_byte = 1048576
    return memory_info.rss / mega_byte  # in bytes


def configure_max_memory_per_process(max_memory_per_process: int | float | None = 4000):
    """Configure the maximum memory to use per process in MB.

    Args:
        max_memory_per_process: The maximum memory to use per process in MB.
    """
    max_memory_per_process = max_memory_per_process if max_memory_per_process is not None else 4000
    logger.info(f"Setting max memory per process to {max_memory_per_process} MB.")
    os.environ["PYSCF_MAX_MEMORY"] = str(max_memory_per_process)
    return max_memory_per_process


def unpack_args_for_imap(func):
    """Decorator to unpack a single tuple argument for a function that is called with imap."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if there is a single argument and it's a tuple for unpacking
        if len(args) == 1 and isinstance(args[0], tuple) and kwargs == {}:
            return func(*args[0])
        else:
            return func(*args, **kwargs)  # Call normally if multiple arguments are passed

    return wrapper
