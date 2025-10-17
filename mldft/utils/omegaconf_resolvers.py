"""OmegaConf resolvers for the mldft package.

The separators of the override in the run names are ``__`` and ``-`` per default (``key1-value1__key2-value2``).
You can change them by setting the environment variables ``MLDFT_RUN_NAME_KEY_VALUE_SEP`` and
``MLDFT_RUN_NAME_OVERRIDE_SEP``, e.g. by adding the following to your shell configuration file:

.. code-block:: bash

    export MLDFT_RUN_NAME_KEY_VALUE_SEP="="
    export MLDFT_RUN_NAME_OVERRIDE_SEP=", "
"""

import os
from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf

from mldft.utils.counter_file import get_and_increment_counter
from mldft.utils.environ import get_mldft_model_path

KEY_VALUE_SEP = os.environ.get("MLDFT_RUN_NAME_KEY_VALUE_SEP", "-")
OVERRIDE_SEP = os.environ.get("MLDFT_RUN_NAME_OVERRIDE_SEP", "__")


def split_overrides_string(overrides_string: str) -> dict:
    """Splits an override string of the form "key1.subkey1=value1,key2=value2,key3/subkey3=value3"
    by commas, which are not in brackets.

    Note: This is still not perfect, e.g. "'" will lead to problems.

    Args:
        overrides_string (str): The override string to split.

    Returns:
        dict: The split override string as a dictionary with keys (before '=') and values (after '=').
    """
    # parse the string and split by ',' which are not inside brackets or quotation marks
    delimiter_map = {"[": "]", "(": ")", "{": "}", '"': '"', "'": "'"}
    all_delimiters = set(delimiter_map.keys()) | set(delimiter_map.values())
    stack = []  # stack of expected closing delimiters
    split = []  # list of splits
    start = 0  # start of the current override
    for i, c in enumerate(overrides_string):
        if c in all_delimiters:
            if c in delimiter_map.values():
                if stack and stack[-1] == c:
                    stack.pop(-1)
                    continue
            if c not in delimiter_map.values() or not stack:
                stack.append(delimiter_map[c])
        elif c == "," and not stack:
            split.append(overrides_string[start:i])
            start = i + 1
    split = split + [overrides_string[start:]]
    result = {override.split("=")[0]: "=".join(override.split("=")[1:]) for override in split}
    return result


def dir_counter(path: str, length: int = 2) -> str:
    """Counts the number of runs in a directory.

    Args:
        path (str): The path to the directory.
        length (int, optional): The length of the counter. Defaults to 2.

    Returns:
        str: The counter as a string with leading zeros.
    """
    if "SLURM_JOB_ID" in os.environ:
        return str(os.environ["SLURM_JOB_ID"].zfill(length))
    else:
        return str(get_and_increment_counter(os.path.join(path, ".run_counter"))).zfill(length)


# register a resolver that counts the number of runs in a directory
OmegaConf.register_new_resolver(
    "dir_counter",
    dir_counter,
)


def create_and_return_dir(path):
    """Creates a directory and returns it."""
    os.makedirs(path, exist_ok=False)
    return path


OmegaConf.register_new_resolver("create_and_return_dir", create_and_return_dir)


def slash_to_backslash(path: str) -> str:
    """Converts slashes to backslashes."""
    return path.replace("/", "\\")


OmegaConf.register_new_resolver("slash_to_backslash", slash_to_backslash)


def _get_leaf_key(full_key: str) -> str:
    """Returns the leaf key of a full key: "key1.subkey1/subsubkey1" -> "subsubkey1" """
    return full_key.split(".")[-1].split("/")[-1]


def leafs_only_override_dirname(overrides_string: str) -> str:
    """Converts an override string of the form
    "key1.subkey1=value1,key2=value2,key3/subkey3=value3" to a more compact form
    "subkey1-value1__key2-value2__subkey3-value3".

    Args:
        overrides_string (str): The override string to compactify, as returned by ``${hydra.job.override_dirname}``.

    Returns:
        str: The compactified override string.
    """
    return OVERRIDE_SEP.join(
        [
            f"{_get_leaf_key(key)}{KEY_VALUE_SEP}{value}"
            for key, value in split_overrides_string(overrides_string).items()
        ]
    )


OmegaConf.register_new_resolver("leafs_only_override_dirname", leafs_only_override_dirname)


def values_only_override_dirname(overrides_string: str) -> str:
    """Converts an override string of the form
    "key1.subkey1=value1,key2=value2,key3/subkey3=value3" to a more compact form
    "value1__value2__value3".

    Args:
        overrides_string (str): The override string to compactify, as returned by ``${hydra.job.override_dirname}``.

    Returns:
        str: The compactified override string.
    """
    return OVERRIDE_SEP.join(split_overrides_string(overrides_string).values())


OmegaConf.register_new_resolver("values_only_override_dirname", values_only_override_dirname)


def checkpoint_path_to_run_number(ckpt_path: str) -> str:
    """Converts a checkpoint path to a run number. E.g. ".../sciai-
    dft/train/runs/448_maybe2_lower_bound_loss/checkpoints/last.ckpt" gets mapped to "448".

    If the checkpoint path is not in the expected format, an exception is raised.

    Args:
        ckpt_path (str): The checkpoint path to convert.

    Returns:
        str: The run number.

    Raises:
        ValueError: If the checkpoint path is not in the expected format.
    """
    try:
        parts = ckpt_path.split("/")
        run_name = parts[-3]
        run_number = run_name.split("_")[0]
        int(run_number)  # make sure it is an integer, but do not return int to keep leading zeros
        return run_number
    except Exception as e:
        raise ValueError(f"Could not convert checkpoint path to run number: {e}")


def smart_override_dirname(overrides_string: str, exclude_keys: list | str = None) -> str:
    """Converts an override string of the form
    "key1.subkey1=str_value1,key2=int_value2,key3/subkey3=float_value3" to a more compact form
    "str_value1__key2-int_value2__subkey3-float_value3", where the subkeys are only shown if the
    value is numeric.

    Args:
        overrides_string (str): The override string to compactify, as returned by ``${hydra.job.override_dirname}``.
        exclude_keys (list | str, optional): A list of keys to exclude from the compactified string. Defaults to None.

    Returns:
        str: The compactified override string.
    """
    # If the length is less than 3, there can't be an override with a = sign
    if len(overrides_string) < 3:
        return ""
    result = []

    if exclude_keys is None:
        exclude_keys = []
    if isinstance(exclude_keys, str):
        exclude_keys = [exclude_keys]

    def is_float_or_bool(value: str) -> bool:
        """Check if the value is a float or a boolean."""
        try:
            float(value)
            return True
        except ValueError:
            return value.lower() in ["true", "false"]

    for full_key, value in split_overrides_string(overrides_string).items():
        # check if key should be excluded
        if full_key in exclude_keys:
            continue
        # don't put the full checkpoint path, it will be too long
        if full_key == "ckpt_path":
            try:
                result.append(f"from_checkpoint_{checkpoint_path_to_run_number(value)}")
            except ValueError:
                logger.warning(
                    f"Could not convert checkpoint path to run number, just adding 'from_checkpoint' to run name. "
                    f"(path: {value})."
                )
                result.append("from_checkpoint")
            continue
        if full_key == "weight_ckpt_path":
            try:
                result.append(f"from_weight_checkpoint_{checkpoint_path_to_run_number(value)}")
            except ValueError:
                logger.warning(
                    f"Could not convert weight checkpoint path to run number, just adding 'from_weight_checkpoint' "
                    f"to run name. (path: {value})."
                )
                result.append("from_weight_checkpoint")
            continue
        # check if value is numeric
        if is_float_or_bool(value):
            result.append(f"{_get_leaf_key(full_key)}{KEY_VALUE_SEP}{value}")
        else:
            result.append(value)
    max_length = os.pathconf(get_mldft_model_path(), "PC_NAME_MAX")
    max_length = max_length - 5  # allows run numbers up to 9999
    return OVERRIDE_SEP.join(result).replace("/", "\\")[:max_length]


OmegaConf.register_new_resolver("smart_override_dirname", smart_override_dirname)


def path_to_basename(path: str) -> str:
    """Converts a path to its basename."""
    if path is None:
        raise ValueError("Provided path is None, did you forget to provide a path?")
    return Path(path).name


OmegaConf.register_new_resolver("path_to_basename", path_to_basename)

# get len:
OmegaConf.register_new_resolver("get_len", lambda x: len(x))


# tensorframes resolvers
def parse_reps_import(rep):
    """Imports and calls the parse_reps function from tensorframes.

    The import is done here to make the tests run without the tensorframes package. If the package
    is not installed, using this resolver will raise an ImportError that indicates that the
    tensorframes package is missing.
    """
    from tensorframes.reps.utils import parse_reps

    return parse_reps(rep)


# getting dim of tensor rep:
OmegaConf.register_new_resolver("get_rep_dim", lambda rep: parse_reps_import(rep).dim)
OmegaConf.register_new_resolver("get_tensorrep_dim", lambda rep: parse_reps_import(f"t{rep}").dim)
OmegaConf.register_new_resolver("get_irrep_dim", lambda rep: parse_reps_import(f"i{rep}").dim)

# parse reps:
OmegaConf.register_new_resolver("parse_reps", lambda rep: parse_reps_import(rep))
