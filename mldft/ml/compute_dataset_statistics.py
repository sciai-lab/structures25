"""Entry point for computing dataset statistics.

The statistics are computed from the training set and saved to disk, at the ``output_dir`` of the run,
as a ``.zarr`` file.
Additionally, if ``extra_save_path`` is specified, the statistics are also saved to that path. By default, this is
set to the location of where the statistics are loaded from in the given training configuration.

By default, no transforms are applied to the dataset.

To run e.g. with local frames, specify the following in ``configs/ml/statistics.yaml``
(or override from any other configuration file):

.. code-block:: yaml

 data:
  datamodule:
    transforms:
     - _target_: mldft.ml.data.components.convert_transforms.ToTorch
     - _target_: mldft.ml.data.components.convert_transforms.AddAtomCooIndices
     - _target_: mldft.ml.data.components.convert_transforms.ToLocalFrames
       matrices: (,)
     - _target_: mldft.ml.data.components.convert_transforms.ToNumpy

Warning:
    Make sure that the transforms are consistent with what you plan to do during training!
"""

from pathlib import Path

import hydra
import lightning as L
import torch
from omegaconf import DictConfig

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.convert_transforms import ProjectGradient
from mldft.ml.data.datamodule import OFDataModule
from mldft.ml.preprocess.dataset_statistics import DatasetStatistics
from mldft.utils import RankedLogger, extras, task_wrapper
from mldft.utils.utils import set_default_torch_dtype

# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/config.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/config.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #


log = RankedLogger(__name__, rank_zero_only=True)


# this import registers custom omegaconf resolvers
import mldft.utils.omegaconf_resolvers  # noqa


@task_wrapper
@set_default_torch_dtype(torch.float64)
def compute_dataset_statistics(cfg: DictConfig):
    """Compute dataset statistics and save them to disk."""

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Due to some statistics taking rather long to compute, we check if the statistics already
    # exist at the extra save path and do not exucute the computation again, unless overwrite is
    # set to true
    extra_save_path = cfg.get("extra_save_path", False)
    overwrite = cfg.get("overwrite", False)
    if extra_save_path:
        extra_save_path = Path(extra_save_path)
        if extra_save_path.exists() and not overwrite:
            raise FileExistsError(
                f"""Dataset statistics already exist at {extra_save_path}. Statistics will not be
                calculated again, unless overwrite: true is set in the config."""
            )

    # This might be needed in the future, if error "too many open files" occurs
    torch.multiprocessing.set_sharing_strategy("file_system")

    log.info(f"Instantiating datamodule <{cfg.data.datamodule._target_}>")
    datamodule: OFDataModule = hydra.utils.instantiate(cfg.data.datamodule)
    assert any(
        isinstance(transform, ProjectGradient)
        for transform in datamodule.transforms.pre_transforms
    ), "Gradients need to be projected for correct statistics."

    # sets up train, val, test datasets
    datamodule.setup(stage="fit")

    basis_info: BasisInfo = datamodule.basis_info

    label_subdir = f"labels_{cfg.data.transforms.name}"
    energy_key = cfg.data.datamodule.dataset_kwargs.get("energy_key", "e_kin")
    out_path = Path(cfg.paths.output_dir) / f"dataset_statistics_{label_subdir}_{energy_key}.zarr"

    dataset_statistics = DatasetStatistics.from_dataset(
        out_path,
        datamodule.train_dataloader(),
        basis_info=basis_info,
        min_atoms_per_type=0,
        **hydra.utils.instantiate(cfg.statistic_fitter_kwargs),
    )

    log.info(f"{dataset_statistics=}")

    # Save to extra save path if specified
    if extra_save_path:
        extra_save_path.parent.mkdir(exist_ok=True)
        try:
            extra_save_path.parent.chmod(0o770)
        except PermissionError:
            log.warning(f"Could not set permissions for {extra_save_path.parent}. ")
        log.info(f"Saving dataset statistics next to the dataset at {extra_save_path}")
        dataset_statistics.save_to(extra_save_path, overwrite=overwrite)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
    }
    metric_dict = {"dataset_statistics": dataset_statistics}
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../../configs/ml", config_name="statistics.yaml")
def main(cfg: DictConfig) -> DatasetStatistics:
    """Entry point for computing dataset statistics.

    Args:
        cfg: DictConfig configuration composed by Hydra.

    Returns:
        Dataset statistics object.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    metric_dict, object_dict = compute_dataset_statistics(cfg)
    statistics = metric_dict["dataset_statistics"]

    return statistics


if __name__ == "__main__":
    main()
