import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def get_path_from_environment_variable(name: str) -> Path:
    """Return the path to the directory specified by the environment variable `name`.

    Args:
        name: The name of the environment variable.

    Returns:
        The path to the directory specified by the environment variable.

    Raises:
        KeyError: If the environment variable is not set.
        AssertionError: If the path does not exist.
    """
    try:
        result = os.environ[name]
    except KeyError:
        raise KeyError(f"{name} environment variable not set.")
    assert os.path.exists(
        result
    ), f'Path "{result}", read from environment variable "{name}", does not exist.'
    return Path(result)


def get_mldft_data_path() -> Path:
    """Return the path to the mldft data directory."""
    return get_path_from_environment_variable("DFT_DATA")


def get_mldft_model_path() -> Path:
    """Return the path to the mldft model directory."""
    return get_path_from_environment_variable("DFT_MODELS")


def get_dataset_from_absolute_path(path: Path) -> str:
    """Return the dataset name from an absolute path."""
    return path.relative_to(get_mldft_data_path()).parts[0]
