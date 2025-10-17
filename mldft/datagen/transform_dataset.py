"""Applying a basis transformation to the dataset.

To convert a dataset to a new basis, this script can be run with:

.. code-block::

    python mldft/datagen/transform_dataset.py data=qm9 data/transforms=local_frames

You can adapt ``data`` and ``data/transforms`` to your needs. The ``use_cached_data`` flag has to be set to ```false```.
"""


import multiprocessing
from collections import defaultdict
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
import zarr
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

import mldft.utils.omegaconf_resolvers  # noqa
from mldft.ml.data.components.basis_transforms import MasterTransformation
from mldft.ml.data.components.convert_transforms import ToNumpy, ToTorch
from mldft.ml.data.components.of_data import OFData
from mldft.utils.environ import get_mldft_data_path
from mldft.utils.multiprocess import configure_processes_and_threads
from mldft.utils.utils import set_default_torch_dtype


def is_valid_label(path: Path, spatial_keys_to_check: list[str]) -> bool:
    """Check if a label file is valid. Does not guarantee that the file is not broken, but checks
    that most of the necessary data is present.

    Args:
        path (Path): Path to the label file.

    Returns:
        bool: True if the label file is valid, False otherwise.
    """
    try:
        with zarr.open(path, mode="r") as root:
            if "of_labels" in root and "ks_labels" in root and "geometry" in root:
                if (
                    "basis" in root["of_labels"]
                    and "n_scf_steps" in root["of_labels"]
                    and "spatial" in root["of_labels"]
                    and "energies" in root["of_labels"]
                ):
                    if all(
                        key in root["of_labels/spatial"].keys() for key in spatial_keys_to_check
                    ):
                        return True
    except Exception as e:
        logger.exception(f"Error while checking {path}: {e}")
    return False


def remove_broken_files(path_list: list[Path], old_label_dir: Path, new_label_dir: Path):
    """Scan the list of zarr files and remove those that are broken, eg. can not be opened or have
    no of_labels.

    Args:
        path_list (list): List of paths to zarr files.
    """
    for path in tqdm(path_list, desc="Searching for broken files", leave=False):
        relative_path = path.relative_to(new_label_dir)
        old_path = old_label_dir / relative_path
        if not is_valid_label(old_path, spatial_keys_to_check=[]):
            raise ValueError(f"No valid labels in old path {old_path}. Consider removing it.")

        # all spatial keys present in the old file should be present in the new file
        root = zarr.open(str(old_path), mode="r")
        old_spatial_keys = root["of_labels/spatial"].keys()
        if not is_valid_label(path, spatial_keys_to_check=old_spatial_keys):
            logger.warning(
                f"Invalid label in {path}, but found valid label in {old_path}. Removing {path}."
            )
            path.unlink()


def convert_zarr_file(args):
    """Convert one label file with the new transforms.

    Args:
        args: Tuple of arguments for the convert function.
    """
    (
        path,
        new_path,
        basis_info,
        transform,
    ) = args
    logger.info(f"Processing {path.name}")
    root = zarr.open(path, mode="r")

    # These checks are not strictly necessary as we also demand the integrals to be present in the data loading,
    # but they are good for piece of mind.
    assert (
        "dual_basis_integrals" in root["of_labels/spatial"]
    ), f"dual_basis_integrals not in {path}. Probably the file was generated with an outdated version of mldft"

    scf_iterations = root["of_labels/n_scf_steps"][()]

    # Copy labels to memory store
    temp_store = zarr.MemoryStore()
    new_root_temp = zarr.open(temp_store, mode="w")

    zarr.copy_all(source=root, dest=new_root_temp)

    spatial_keys = root["of_labels/spatial"].keys()

    # Delete copied spatial labels, to make sure none are incorrectly left untransformed
    of_spatial = new_root_temp.create_group("of_labels/spatial", overwrite=True)

    new_of_spatial = defaultdict(list)
    for i in range(scf_iterations):
        # Load sample, apply transforms, save in list
        sample = OFData.from_file_with_all_gradients(path, i, basis_info)
        sample = transform(sample)
        for key in spatial_keys:
            new_of_spatial[key].append(sample[key])
    # Stack the lists to tensors and save them
    new_of_spatial = {key: np.stack(value) for key, value in new_of_spatial.items()}
    for key, value in new_of_spatial.items():
        of_spatial.create_dataset(key, data=value, chunks=(1, value.shape[-1]), dtype=np.float64)

    # Create new zip store and move everything from the temp store to the new store
    # This should be much better to prevent too many broken files
    with zarr.ZipStore(new_path, mode="w") as new_store:
        new_root = zarr.open(new_store, mode="w")
        zarr.copy_all(source=new_root_temp, dest=new_root)
    temp_store.clear()


@set_default_torch_dtype(torch.float64)
def transform_dataset(cfg: DictConfig) -> None:
    """Script to basis transform the dataset.

    Applies the given transform to the dataset and saves the new labels in a new folder.

    Args:
        cfg (DictConfig): Config, see `configs/ml/train.yaml`.
    """
    num_processes, num_threads_per_process = configure_processes_and_threads(
        cfg.get("num_processes"), cfg.get("num_threads_per_process")
    )
    logger.remove()
    logger_format = (
        "<green>{time:HH:mm:ss}</green>|<level>{level: <8}</level>|<level>{message}</level>"
    )
    logger.add(
        lambda msg: tqdm.write(msg, end=""), format=logger_format, colorize=True, enqueue=True
    )
    # Configure transforms
    data_path = get_mldft_data_path()
    dataset_path = data_path / cfg.data.dataset_name
    basis_info = hydra.utils.instantiate(cfg.data.basis_info)
    cached_transforms = hydra.utils.instantiate(cfg.data.transforms.cached_transforms)
    master_transform = MasterTransformation(
        name=cfg.data.transforms.cached_transforms.name,
        use_cached_data=False,
        pre_transforms=[ToTorch(float_dtype=torch.float64)],
        cached_transforms=cached_transforms,
        post_transforms=[ToNumpy()],
    )
    old_label_dir = dataset_path / master_transform.label_subdir
    logger.info(f"Applying transform: {master_transform.name}")
    # This file handling here is not nice but good enough
    new_label_dir = old_label_dir.parent / f"labels_{master_transform.name}"
    new_label_dir.mkdir(exist_ok=True)
    try:
        new_label_dir.chmod(0o770)
    except PermissionError:
        logger.warning(f"Could not change permissions of {new_label_dir}.")
    path_list = list(old_label_dir.rglob("*.zarr*"))

    start_idx = cfg.get("start_idx", 0)
    num_molecules = cfg.get("num_molecules", None)
    if num_molecules is None:
        num_molecules = len(path_list)
    if start_idx + num_molecules > len(path_list):
        end_idx = len(path_list)
    else:
        end_idx = start_idx + num_molecules
    path_list = sorted(path_list)[start_idx:end_idx]

    paths = set(path_list)
    new_paths_list = list(sorted(new_label_dir.rglob("*.zarr*")))
    # Check for broken files and delete them
    remove_broken_files(new_paths_list, old_label_dir, new_label_dir)
    # Scan again after deleting broken files
    new_paths_list = list(new_label_dir.rglob("*.zarr*"))
    for zarr_file in tqdm(new_paths_list, desc="Identifying already converted files", leave=False):
        old_name = old_label_dir / zarr_file.relative_to(new_label_dir)
        if old_name in paths:
            paths.remove(old_name)
    args_list = [
        (path, new_label_dir / path.relative_to(old_label_dir), basis_info, master_transform)
        for path in sorted(paths)
    ]
    if len(args_list) > 0:
        with multiprocessing.Pool(num_processes) as pool:
            list(
                tqdm(
                    pool.imap_unordered(convert_zarr_file, args_list),
                    total=len(args_list),
                    position=0,
                    smoothing=0,
                    dynamic_ncols=True,
                    desc="Converting labels",
                )
            )
    logger.info("Finished transforming dataset!")
    return


@hydra.main(version_base="1.3", config_path="../../configs/ml", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Hydra entry point for the script."""
    transform_dataset(cfg)


if __name__ == "__main__":
    main()
