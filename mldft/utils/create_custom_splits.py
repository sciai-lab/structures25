import argparse
import pickle
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from tqdm import tqdm

from mldft.utils.create_dataset_splits import (
    _create_train_val_test_split_dict,
    _iteration_count,
    check_paths_and_save,
    get_path_list_for_split_file,
)
from mldft.utils.environ import get_mldft_data_path


def qmugs_dataset_to_csv(name: str):
    dataset_name = name.split("_")[0]
    return f"{dataset_name}.csv"


def get_qmugs_paths_and_group_ids(
    qmugs_path: Path, csv_file: Path
) -> tuple[list[Path], list[int]]:
    """Determine paths of QMUGS label files based on the id given in the csv file."""
    # To determine which QMUGS files belong to this dataset, we need to read the ids from the csv file
    assert csv_file.exists(), f"CSV file {csv_file} does not exist."
    dataframe = pl.read_csv(csv_file)
    ids = dataframe["id"]
    paths = []
    group_ids = []
    label_dir = qmugs_path / "labels"
    label_files = list(label_dir.glob("*.zarr*"))
    for label in tqdm(label_files, desc="Identifying qmugs paths and chembl ids"):
        if int(label.name.split(".")[0]) in ids:
            paths.append(label)
            group_ids.append(int(label.name.split(".")[0]) // 3)
    return paths, group_ids


def create_split_file_qmugs_large_bins(dataset: str, override: bool):
    """Create a split file containing only validation molecules for the QMUGSLargeBins dataset.

    Train and test splits are empty.
    """
    logger.info(f"Creating QMUGS large bins split file for {dataset}.")
    data_dir = get_mldft_data_path()
    qmugs_base_path = data_dir / "QMUGS"
    qmugs_large_bins_path = data_dir / "QMUGSLargeBins"
    # To determine which QMUGS files belong to this dataset, we need to read the ids from the csv file
    csv_file = qmugs_base_path / qmugs_dataset_to_csv(dataset)
    paths, groups = get_qmugs_paths_and_group_ids(qmugs_base_path, csv_file)
    iterations = []
    for path in tqdm(paths, desc="Creating Dataset, counting scf iterations per path"):
        iterations.append(_iteration_count(path))
    iterations = np.array(iterations)
    paths_train = []
    train_size = 0
    paths_val = []
    val_size = 0
    paths_test = get_path_list_for_split_file(paths, iterations)
    test_size = int(sum(iterations))
    yaml_dict = {
        "sizes": {"train": train_size, "val": val_size, "test": test_size},
        "train": paths_train,
        "val": paths_val,
        "test": paths_test,
    }
    yaml_path = qmugs_large_bins_path / "split.yaml"
    pickle_path = qmugs_large_bins_path / "split.pkl"
    check_paths_and_save(yaml_dict, yaml_path, pickle_path, override)


def create_split_file_qmugs_bin(dataset: str, override: bool):
    """Create a split file containing the first bin of QMUGS."""
    logger.info(f"Creating QMUGS bin split file for {dataset}.")
    data_dir = get_mldft_data_path()
    if "perturbed_fock" in dataset:
        qmugs_base_path = data_dir / "QMUGS_perturbed_fock"
    else:
        qmugs_base_path = data_dir / "QMUGS"
    qmugs_csv_dir = data_dir / "QMUGS"
    csv_file = qmugs_csv_dir / qmugs_dataset_to_csv(dataset)
    logger.info(f"Reading molecule ids from csv file {csv_file}.")
    paths, group_ids = get_qmugs_paths_and_group_ids(qmugs_base_path, csv_file)
    qmugs_bin_splits = _create_train_val_test_split_dict(
        paths, group_ids, (0.9, 0.1, 0), processes=1
    )
    large_bin_split_file = data_dir / "QMUGSLargeBins" / "split.pkl"
    # If the large mol split file exists, use it for the test split
    if large_bin_split_file.exists():
        with open(large_bin_split_file, "rb") as f:
            large_bin_splits = pickle.load(f)
        qmugs_bin_splits["test"] = large_bin_splits["test"]
        qmugs_bin_splits["sizes"]["test"] = large_bin_splits["sizes"]["test"]
    else:
        logger.warning(
            f"Large bin split file {large_bin_split_file} does not exist. Test split will be empty."
        )
    yaml_path = data_dir / dataset / "split.yaml"
    pickle_path = data_dir / dataset / "split.pkl"
    check_paths_and_save(qmugs_bin_splits, yaml_path, pickle_path, override)


def create_split_file_qmugs_bin0_qm9(dataset: str, override: bool):
    """Create a split file containing qm9 train set + the first bin of QMUGS."""
    logger.info(f"Creating QMUGS and QM9 merged split file for {dataset}.")
    data_dir = get_mldft_data_path()
    with open(data_dir / "QMUGSLargeBins" / "split.pkl", "rb") as f:
        qmugs_large_bins_splits = pickle.load(f)
    if "perturbed_fock" in dataset:
        qmugs_bin0_path = data_dir / "QMUGSBin0_perturbed_fock"
        with open(qmugs_bin0_path / "split.pkl", "rb") as f:
            qmugs_first_bin_splits = pickle.load(f)
        with open(data_dir / "QM9_perturbed_fock" / "split.pkl", "rb") as f:
            qm9_splits = pickle.load(f)
        merged_path = data_dir / "QMUGSBin0QM9_perturbed_fock"
    else:
        qmugs_bin0_path = data_dir / "QMUGSBin0"
        with open(qmugs_bin0_path / "split.pkl", "rb") as f:
            qmugs_first_bin_splits = pickle.load(f)
        with open(data_dir / "QM9_mof" / "split.pkl", "rb") as f:
            qm9_splits = pickle.load(f)
        merged_path = data_dir / "QMUGSBin0QM9"

    # Merge splits
    qmugs_first_bin_splits["train"] += qm9_splits["train"] + qm9_splits["val"]
    qmugs_first_bin_splits["val"] += qm9_splits["test"]
    qmugs_first_bin_splits["sizes"]["train"] += (
        qm9_splits["sizes"]["train"] + qm9_splits["sizes"]["val"]
    )
    qmugs_first_bin_splits["sizes"]["val"] += qm9_splits["sizes"]["test"]
    qmugs_first_bin_splits["test"] = qmugs_large_bins_splits["test"]
    qmugs_first_bin_splits["sizes"]["test"] = qmugs_large_bins_splits["sizes"]["test"]
    yaml_path = merged_path / "split.yaml"
    pickle_path = merged_path / "split.pkl"
    check_paths_and_save(qmugs_first_bin_splits, yaml_path, pickle_path, override)


if __name__ == "__main__":
    # parse label_dir from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Name of the dataset.")
    parser.add_argument("--override", action="store_true", help="Override existing split file.")
    args = parser.parse_args()
    if args.dataset == "QMUGSLargeBins":
        create_split_file_qmugs_large_bins(args.dataset, args.override)
    elif args.dataset.startswith("QMUGSBin0QM9"):
        create_split_file_qmugs_bin0_qm9(args.dataset, args.override)
    elif args.dataset.startswith("QMUGSBin"):
        create_split_file_qmugs_bin(args.dataset, args.override)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")
