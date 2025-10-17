from typing import List

import hydra
import torch
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, open_dict

from mldft.ml.data.components.of_data import Representation
from mldft.ml.data.datamodule import OFDataModule
from mldft.ml.models.mldft_module import MLDFTLitModule
from mldft.utils.log_utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    Args:
        callbacks_cfg: A DictConfig object containing callback configurations.

    Returns:
        A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config.

    Args:
        logger_cfg: A DictConfig object containing logger configurations.

    Returns:
        A list of instantiated loggers.
    """
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


def instantiate_model(
    checkpoint_path,
    device: str | torch.device,
    model_dtype: torch.dtype = torch.float64,
) -> MLDFTLitModule:
    """Instantiate a model from a checkpoint.

    Args:
        checkpoint_path: The path to the checkpoint.
        device: The device to load the model on.
        model_dtype: The dtype of the model.
        deterministic: Whether the model should be deterministic.

    Returns:
        The instantiated model.
    """
    lightning_module = MLDFTLitModule.load_from_checkpoint(checkpoint_path, map_location=device)
    lightning_module.eval()
    lightning_module.to(model_dtype)
    return lightning_module


def instantiate_datamodule(
    cfg: DictConfig, limit_scf_iterations: int | list[int] | None = -1
) -> OFDataModule:
    """Instantiates datamodule from config.

    Instantiates a datamodule from the provided configuration. Adds additional keys and the
    transformation matrix.

    Args:
        cfg: A DictConfig object containing the train configuration.
        limit_scf_iterations: Which SCF iterations to use (see
            :py:class:`mldft.ml.data.components.dataset.OFDataset`). By default, only the ground
            state is loaded.

    Returns:
        An instantiated datamodule.
    """
    # copy the datamodule config to avoid side effects
    cfg = cfg.copy()
    datamodule = cfg.data.datamodule
    with open_dict(datamodule):
        datamodule.batch_size = 1
        datamodule.num_workers = 1
        datamodule.transforms.add_transformation_matrix = True
        datamodule.transforms.use_cached_data = False
        datamodule.transforms.pre_transforms = [
            {
                "_target_": "mldft.ml.data.components.convert_transforms.AddOverlapMatrix",
                "basis_info": datamodule.basis_info,
            },
        ] + datamodule.transforms.pre_transforms
        datamodule.transforms.post_transforms = []
        datamodule.dataset_kwargs.limit_scf_iterations = limit_scf_iterations
        datamodule.dataset_kwargs.keep_initial_guess = False
        datamodule.dataset_kwargs.additional_keys_at_scf_iteration = {
            "of_labels/spatial/grad_kin": Representation.GRADIENT,
            "of_labels/spatial/grad_xc": Representation.GRADIENT,
        }
        datamodule.dataset_kwargs.additional_keys_at_ground_state = {
            "of_labels/spatial/grad_kin": Representation.GRADIENT,
            "of_labels/spatial/grad_xc": Representation.GRADIENT,
            "of_labels/energies/e_electron": Representation.SCALAR,
            "of_labels/energies/e_ext": Representation.SCALAR,
            "of_labels/energies/e_hartree": Representation.SCALAR,
            "of_labels/energies/e_kin": Representation.SCALAR,
            "of_labels/energies/e_kin_plus_xc": Representation.SCALAR,
            "of_labels/energies/e_kin_minus_apbe": Representation.SCALAR,
            "of_labels/energies/e_kinapbe": Representation.SCALAR,
            "of_labels/energies/e_xc": Representation.SCALAR,
            "of_labels/energies/e_tot": Representation.SCALAR,
        }
        datamodule.dataset_kwargs.additional_keys_per_geometry = {
            "of_labels/n_scf_steps": Representation.NONE,
        }

    return hydra.utils.instantiate(datamodule)
