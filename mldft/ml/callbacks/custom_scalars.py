from collections import defaultdict

import lightning as pl
from lightning import Callback
from loguru import logger

from mldft.ml.models.components.loss_function import WeightedLoss


class AddMetricAndLossCustomScalars(Callback):
    """Add custom scalars to the tensorboard logger, for comparison of train and val metrics and
    losses."""

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Adds the custom scalars."""
        tb_logger = pl_module.tensorboard_logger
        if tb_logger is None:
            logger.warning("No tensorboard logger found. Skipping addition of custom scalars.")
            return

        def _get_group(metric_key):
            """e.g. 'train_metrics_per_electron/total_loss' -> 'metrics per electron'."""
            return metric_key.split("/")[0].replace("_", " ").replace("train", "").strip()

        layout = defaultdict(dict)
        # metrics
        for key, val in pl_module.train_metrics.items():
            group = _get_group(key)
            layout[group][key.split("/")[-1]] = ["Multiline", [key, key.replace("train", "val")]]

        # losses
        layout["losses"]["total"] = ["Multiline", ["train_loss/total", "val_loss/total"]]
        if isinstance(pl_module.loss_function, WeightedLoss):
            for key, weight in pl_module.loss_function.weight_dict.items():
                layout["losses"][key] = ["Multiline", [f"train_loss/{key}", f"val_loss/{key}"]]
        # layout['losses']['all'] = ['Multiline', ['train_loss/*', 'val_loss/*']]

        tb_logger.add_custom_scalars(layout)
