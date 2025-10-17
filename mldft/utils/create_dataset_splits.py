"""These scripts are not super tidy, but they only need to be reproducible."""

import argparse
import multiprocessing
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
import zarr
from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from mldft.utils.environ import get_dataset_from_absolute_path, get_mldft_data_path


def _iteration_count(path: Path) -> int:
    """Opens a zarr file and returns the scf iterations saved in it.

    Args:
        path: the path to the zarr file.

    Returns:
        The number of scf iteration
    """
    with zarr.open(path, mode="r") as root:
        if "n_scf_steps" in root["of_labels"]:
            return root["of_labels/n_scf_steps"][()]
        else:
            raise KeyError(f" Key n_scf_steps not found in {path}")


def split_non_grouped(ids: np.ndarray, split_percentages: tuple[float, float, float]) -> tuple:
    """Split ids into train, val and test sets based on the split percentages."""
    total_size = len(ids)
    # Adjust split percentages if dataset is too small
    min_required = 1
    create_test_split = True
    if split_percentages[2] * total_size < min_required:
        split_percentages = (
            split_percentages[0],
            split_percentages[1] + split_percentages[2],
            0,
        )
        logger.warning(
            f"Test split is too small, adjusting split percentages to {split_percentages}."
        )
        create_test_split = False
    if not create_test_split:  # Train and validation split
        ids_train, ids_val = train_test_split(
            ids,
            train_size=split_percentages[0],
            test_size=split_percentages[1],
            shuffle=True,
            random_state=8,
        )
        ids_test = []
    else:
        ids_train, ids_val_test = train_test_split(
            ids,
            train_size=split_percentages[0],
            test_size=split_percentages[1] + split_percentages[2],
            shuffle=True,
            random_state=8,
        )
        ids_val, ids_test = train_test_split(
            ids_val_test,
            train_size=split_percentages[1] / (split_percentages[1] + split_percentages[2]),
            test_size=split_percentages[2] / (split_percentages[1] + split_percentages[2]),
            shuffle=True,
            random_state=8,
        )
    return ids_train, ids_val, ids_test


def _train_val_test_split(
    ids: np.ndarray, split_percentages: tuple[float, float, float]
) -> tuple[list, list, list]:
    """Split the ids into train, val and test sets."""
    n_total = len(ids)
    n_test = int(n_total * split_percentages[2])
    n_val = int(n_total * split_percentages[1])
    n_train = n_total - n_val - n_test
    np.random.seed(8)
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]
    return (ids[train_indices].tolist(), ids[val_indices].tolist(), ids[test_indices].tolist())


def split_grouped(
    group_ids: np.ndarray, split_percentages: tuple[float, float, float]
) -> tuple[list, list, list]:
    group_id_to_indices = defaultdict(list)
    for i, group_id in enumerate(group_ids):
        group_id_to_indices[group_id].append(i)
    unique_group_ids = np.array(list(group_id_to_indices.keys()))
    group_ids_train, group_ids_val, group_ids_test = _train_val_test_split(
        unique_group_ids, split_percentages
    )
    # Collect indices for each split
    indices_train = []
    for group_id in group_ids_train:
        indices_train.extend(group_id_to_indices[group_id])
    indices_val = []
    for group_id in group_ids_val:
        indices_val.extend(group_id_to_indices[group_id])
    indices_test = []
    for group_id in group_ids_test:
        indices_test.extend(group_id_to_indices[group_id])
    return indices_train, indices_val, indices_test


def get_path_list_for_split_file(paths: list[Path], iterations: np.ndarray) -> list[list]:
    """Turn list of paths and iterations into required format of dataset, name, iterations."""
    return [
        [
            get_dataset_from_absolute_path(path),
            path.name,
            int(iteration),
        ]
        for path, iteration in zip(paths, iterations)
    ]


def check_paths_and_save(
    yaml_dict: dict, yaml_path: Path, pickle_path: Path, override: bool = False
):
    """Check if split file already exists, if so, check if it is the same.

    If it's the same or override is False, return. else, save the new split file to the given
    paths.
    """
    if yaml_path.exists():
        logger.warning(f"Split file {yaml_path} already exists.")
        old_yaml_dict = yaml.safe_load(yaml_path.open("r"))
        if old_yaml_dict == yaml_dict:
            logger.info("The new split file is the same as the old one.")
        elif not override:
            logger.warning(
                "The new split file is different from the old one. The old one will not be overwritten. "
                "If you want to override the old one, use the --override flag."
            )
            return
        else:
            logger.warning(
                "The new split file is different from the old one. The old one will be overwritten."
            )
    logger.info(f"Saving split file to {yaml_path}.")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w") as f:
        yaml.dump(yaml_dict, f, sort_keys=False, default_flow_style=None)
    yaml_path.chmod(0o770)
    logger.info(f"Saving split file to {pickle_path}.")
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with pickle_path.open("wb") as f:
        pickle.dump(yaml_dict, f, pickle.HIGHEST_PROTOCOL)
    pickle_path.chmod(0o770)


def _create_train_val_test_split_dict(
    paths: list[Path],
    group_ids: np.ndarray[int] | list[int] | None = None,
    split_percentages: tuple[float, float, float] = (0.8, 0.1, 0.1),
    processes: int = 1,
    all_test: bool = False,
):
    """Create a train, val, test split from the dataset.

    Args:
        paths: List of paths to the files containing .zarr labels.
        ids: List of ids to use for the split. If None, the ids are generated from the paths.
        split_percentages: Percentages of the dataset to use for train, val and test.
        processes: Number of processes to use for counting scf iterations.

    Returns:
        train, val, test split.
    """
    assert len(paths) > 0, "No paths given."
    logger.info(f"Using {processes} processes")
    if processes > 1:
        with multiprocessing.Pool(processes=processes) as pool:
            iterations = list(
                tqdm(
                    pool.imap(
                        _iteration_count,
                        paths,
                        chunksize=100,
                    ),
                    total=len(paths),
                    desc="Creating Dataset, counting scf iterations per path",
                )
            )
    else:
        iterations = [
            _iteration_count(path)
            for path in tqdm(paths, desc="Creating Dataset, counting scf iterations per path")
        ]
    iterations = np.array(iterations)
    if not all_test:
        if group_ids is None:
            indices_train, indices_val, indices_test = split_non_grouped(
                np.arange(len(paths)), split_percentages
            )
        else:
            indices_train, indices_val, indices_test = split_grouped(group_ids, split_percentages)
    else:
        indices_train, indices_val, indices_test = [], [], np.arange(len(paths))
    paths_train = [paths[i] for i in indices_train]
    iterations_train = iterations[indices_train]
    paths_val = [paths[i] for i in indices_val]
    iterations_val = iterations[indices_val]
    paths_test = [paths[i] for i in indices_test]
    iterations_test = iterations[indices_test]
    # Compute sizes, convert paths to strings and save to yaml
    # Need to call int because yaml does not support numpy integers
    train_size = int(sum(iterations_train))
    val_size = int(sum(iterations_val))
    test_size = int(sum(iterations_test))
    # Sort paths and iterations
    if len(paths_train) > 0:
        paths_train, iterations_train = zip(
            *sorted(zip(paths_train, iterations_train), key=lambda x: x[0])
        )
    if len(paths_val) > 0:
        paths_val, iterations_val = zip(
            *sorted(zip(paths_val, iterations_val), key=lambda x: x[0])
        )
    if len(paths_test) > 0:
        paths_test, iterations_test = zip(
            *sorted(zip(paths_test, iterations_test), key=lambda x: x[0])
        )
    logger.info(f"Train size: {train_size}, val size: {val_size}, test size: {test_size}")
    paths_train = get_path_list_for_split_file(paths_train, iterations_train)
    paths_val = get_path_list_for_split_file(paths_val, iterations_val)
    paths_test = get_path_list_for_split_file(paths_test, iterations_test)
    yaml_dict = {
        "sizes": {"train": train_size, "val": val_size, "test": test_size},
        "train": paths_train,
        "val": paths_val,
        "test": paths_test,
    }
    return yaml_dict


def create_split_file(
    dataset: str,
    yaml_path: str | Path | None,
    pickle_path: str | Path | None,
    override: bool,
    split_percentages: tuple[float, float, float],
    processes: int,
    all_test: bool = False,
):
    """Create a split file for any dataset by reading the zarr files inside the labels
    directory."""
    label_dir = get_mldft_data_path() / dataset / "labels"
    assert label_dir.exists(), f"Label directory {label_dir} does not exist."
    paths = sorted(label_dir.glob("*.zarr*"))
    if len(paths) == 0:
        if len(os.listdir(label_dir)) == 1:
            paths = sorted(label_dir.rglob("*.zarr*"))
        else:
            raise FileNotFoundError(
                f"There are no .zarr files in {label_dir} and an ambiguous amount of subfolders."
            )
    yaml_dict = _create_train_val_test_split_dict(
        paths,
        split_percentages=split_percentages,
        processes=processes,
        all_test=all_test,
    )
    dataset_dir = get_mldft_data_path() / dataset
    if yaml_path is None:
        yaml_path = dataset_dir / "split.yaml"
    if pickle_path is None:
        pickle_path = dataset_dir / "split.pkl"
    check_paths_and_save(yaml_dict, yaml_path, pickle_path, override)


if __name__ == "__main__":
    # parse label_dir from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Name of the dataset.")
    parser.add_argument(
        "--split_percentages",
        nargs=3,
        type=float,
        default=(0.8, 0.1, 0.1),
        help="Percentages of the dataset to use for train, val and test.",
    )
    parser.add_argument("--override", action="store_true", help="Override existing split file.")
    parser.add_argument("--yaml_path", type=Path, help="Path to the yaml file.")
    parser.add_argument("--pickle_path", type=Path, help="Path to the pickle file.")
    parser.add_argument(
        "--all-test",
        action="store_true",
        help="Use all data for test. split_percentages will be ignored.",
    )
    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        help="Number of processes to use, defaults to all available",
        default=multiprocessing.cpu_count(),
    )
    args = parser.parse_args()
    create_split_file(
        args.dataset,
        args.yaml_path,
        args.pickle_path,
        args.override,
        args.split_percentages,
        args.processes,
        args.all_test,
    )
