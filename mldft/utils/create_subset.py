import argparse
import pickle
import shutil
from pathlib import Path

import numpy as np
import yaml
from loguru import logger

from mldft.utils.environ import get_mldft_data_path


def create_subset(
    dataset_path: Path,
    subset_path: Path,
    label_dirs: list,
    split_file_name: str,
    reduction: int | float = 100,
):
    """Script to create a subset of a dataset.

    Args:
        dataset_path (Path): Path to the dataset to create a subset of.
        subset_path (Path): Path to the subset directory.
        label_dirs (list): List of label directories in the dataset, from these the labels will be copied.
        split_file_name (Path): Name of the split file.
        reduction (int, optional): Factor to reduce the dataset size by. Defaults to 100.
    """
    with open(dataset_path / f"{split_file_name}.pkl", "rb") as f:
        split = pickle.load(f)
    new_splits = {}
    train_val_test = ["train", "val", "test"]
    for key in train_val_test:
        samples = split[key]
        new_splits[key] = []
        n_samples = len(samples)
        logger.info(f"Old {key} set has {n_samples} geometries.")
        generator = np.random.Generator(np.random.PCG64(1))
        new_size = int(n_samples / reduction)
        logger.info(f"Sampling {new_size} of those {n_samples} geometries.")
        new_sample_ids = generator.choice(np.arange(n_samples), size=new_size, replace=False)
        new_samples = [samples[i] for i in new_sample_ids]
        for sample in new_samples:
            for dir in label_dirs:
                label_path = dataset_path / dir / sample[1]
                if label_path.exists():
                    new_label_path = subset_path / dir / sample[1]
                    new_label_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(label_path, new_label_path)
                    new_splits[key].append(sample)
                else:
                    print(f"{label_path} does not exist")
        logger.info(f"New {key} set has {len(new_splits[key])} geometries.")

    logger.info("Saving split file.")
    new_split_yaml = subset_path / split_file_name
    with new_split_yaml.open("w") as f:
        yaml.dump(new_splits, f, sort_keys=False, default_flow_style=None)
    try:
        new_split_yaml.chmod(0o770)
    except Exception as e:
        logger.warning(f"Could not change permissions of {new_split_yaml}: {e}")
    pickle_path = subset_path / f"{split_file_name}.pkl"
    with pickle_path.open("wb") as f:
        pickle.dump(new_splits, f, pickle.HIGHEST_PROTOCOL)
    try:
        pickle_path.chmod(0o770)
    except Exception as e:
        logger.warning(f"Could not change permissions of {pickle_path}: {e}")
    if (dataset_path / "basis_transformations").exists():
        shutil.copytree(
            dataset_path / "basis_transformations", subset_path / "basis_transformations"
        )
        logger.info("Copied basis transformations.")
    else:
        logger.warning("No basis transformations found.")
    if (dataset_path / "dataset_statistics").exists():
        shutil.copytree(dataset_path / "dataset_statistics", subset_path / "dataset_statistics")
        logger.info("Copied dataset statistics.")
    else:
        logger.warning("No dataset statistics found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Name of the dataset to create a subset of.")
    parser.add_argument(
        "--reduction", type=float, default=100, help="Factor to reduce the dataset size by."
    )
    parser.add_argument(
        "--override", action="store_true", help="Whether to override the existing subset."
    )
    args = parser.parse_args()
    data_path = get_mldft_data_path()
    dataset_path = data_path / args.dataset
    subset_path = data_path / f"{args.dataset}subset"
    if subset_path.exists() and not args.override:
        raise FileExistsError(f"Subset {subset_path} already exists.")
    logger.info(f"Creating subset of {args.dataset} dataset at {subset_path}.")
    logger.info(f"Reducing dataset size by factor {args.reduction}.")
    label_dirs = sorted(dir.name for dir in dataset_path.glob("labels*") if dir.is_dir())
    split_str = "train_val_test_split"
    create_subset(dataset_path, subset_path, label_dirs, split_str, args.reduction)
