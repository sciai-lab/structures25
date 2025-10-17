"""Main entry point for training."""

from pathlib import Path
from random import randint
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf, open_dict

# this import registers custom omegaconf resolvers
import mldft.utils.omegaconf_resolvers  # noqa
from mldft.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    log_hyperparameters,
    task_wrapper,
)
from mldft.utils.instantiators import instantiate_callbacks, instantiate_loggers
from mldft.utils.log_utils.config_in_tensorboard import log_config_text_to_tensorboard

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    if (ckpt_path := cfg.get("ckpt_path")) is not None:
        if cfg.get("use_original_settings"):
            log.info("Using original settings from checkpoint!")
            config_path = Path(ckpt_path).parent.parent / "hparams.yaml"
            cfg = OmegaConf.load(config_path)

    # Save config as unresolved yaml for checkpoint loading later
    output_dir = Path(cfg.paths.output_dir)
    yaml_path = output_dir / "hparams.yaml"
    yaml_path_resolved = output_dir / "hparams_resolved.yaml"
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    else:
        seed = randint(0, 2**32 - 1)
        L.seed_everything(seed, workers=True)
        with open_dict(cfg):
            cfg.seed = seed
    log.info(f"Logging config to {yaml_path}")
    OmegaConf.save(cfg, yaml_path, resolve=False)
    OmegaConf.save(cfg, yaml_path_resolved, resolve=True)
    # This might be needed in the future, if error "too many open files" occurs
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_float32_matmul_precision("high")
    # If precision is 64, set default dtype to float64 which changes the dtype in the last to_torch transform
    if cfg.trainer.precision == 64:
        torch.set_default_dtype(torch.float64)

    log.info(f"Instantiating datamodule <{cfg.data.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    # log config to tensorboard as text
    log_config_text_to_tensorboard(cfg, trainer)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }
    # If there is no hparams.yaml already this saves a hparams.yaml from lightning logger, but we have one already.
    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if (weight_ckpt_path := cfg.get("weight_ckpt_path")) is not None:
        assert (
            ckpt_path is None
        ), "Cannot load both model and weights from checkpoint! Model weights would be overwritten."
        model.load_state_dict(torch.load(weight_ckpt_path, map_location="cpu")["state_dict"])

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    train_metrics = trainer.callback_metrics

    if cfg.get("validate"):
        log.info("Starting validation!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    val_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **val_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../../configs/ml", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
