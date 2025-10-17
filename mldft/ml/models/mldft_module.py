"""Lightning module for the neural network."""

from typing import Any, Dict, Tuple

import torch
import torch_geometric
from lightning import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from torch_geometric.data import Batch
from torchmetrics import MeanMetric, MinMetric

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.of_data import OFData, Representation
from mldft.ml.models.components.loss_function import project_gradient_difference
from mldft.ml.models.components.sample_weighers import GroundStateOnlySampleWeigher
from mldft.ml.models.components.training_metrics import (
    MAEEnergy,
    MAEGradient,
    MAEInitialGuess,
)
from mldft.utils import RankedLogger
from mldft.utils.log_utils.logging_mixin import locate_logging_mixins


def get_layers(model: torch.nn.Module):
    children = list(model.children())
    return [model] if len(children) == 0 else [ci for c in children for ci in get_layers(c)]


class MLDFTLitModule(LightningModule):
    """The MLDFTLitModule class is a LightningModule that is used to wrap the neural network in the
    pytorch lightning framework."""

    def __init__(
        self,
        net: torch.nn.Module,
        basis_info: BasisInfo,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        target_key: str,
        loss_function: torch.nn.Module,
        compile: bool,
        variational: bool = True,
        metric_interval: int = 10,
        logging_mixin_interval: int | None = None,
        show_logging_mixins_in_progress_bar: bool = False,
    ) -> None:
        """Initializes the MLDFTLitModule object.

        Args:
            net (torch.nn.Module): the neural network
            basis_info (BasisInfo): the basis info object used for the dataset. Added here for logging purposes.
            optimizer (torch.optim.Optimizer): the optimizer to use for training
            scheduler (torch.optim.lr_scheduler): the learning rate scheduler to use for training
            loss_function (torch.nn.Module): the loss function to use for training
            target_key (str): the name of the target key. Added here to easily determine which
                targets were used for training afterward.
            compile (bool): whether to compile the model with :func:`torch.compile`
            variational (bool): whether the model is variational or not. If True, the model is assumed to predict
                two outputs, the energy and the coefficient difference to the ground state (for proj minao).
                If False, the model is assumed to predict the gradient directly, in a non-variational manner, so
                three outputs are expected: The energy, the gradient and the coefficient difference.
            metric_interval (int): the interval (in steps) at which the metrics are calculated and logged.
            logging_mixin_interval (int | None): the interval (in steps) at which the logging mixins are called.
                Defaults to None, which means that the logging mixins are not called.
            show_logging_mixins_in_progress_bar (bool): whether to show the values logged using the logging mixins in
                the progress bar. Defaults to False.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # logger=false means that the hyperparameters will not be logged
        self.save_hyperparameters(logger=False)

        # save the neural network
        self.net = net
        self.target_key = target_key
        self.variational = variational

        # define loss function
        self.loss_function = loss_function
        self.console_logger = RankedLogger(__name__, rank_zero_only=True)

        # metric objects for calculating and averaging accuracy across batches

        # The metric objects have to be initialized in this way, otherwise Lighting doesn't recognize them
        # Dicts of the metrics to iterate through when logging
        def create_metric_module_dict(split: str):
            """Creates a dictionary of metrics for a given split."""
            return torch.nn.ModuleDict(
                {
                    f"{split}_metrics/mae_energy": MAEEnergy(mode="per molecule"),
                    f"{split}_metrics_ground_state/mae_energy_ground_state": MAEEnergy(
                        mode="per molecule", sample_weigher=GroundStateOnlySampleWeigher()
                    ),
                    f"{split}_metrics_per_electron/mae_energy_per_electron": MAEEnergy(
                        mode="per electron"
                    ),
                    f"{split}_metrics/mae_gradient": MAEGradient(mode="per molecule"),
                    f"{split}_metrics_ground_state/mae_gradient_ground_state": MAEGradient(
                        mode="per molecule", sample_weigher=GroundStateOnlySampleWeigher()
                    ),
                    f"{split}_metrics_per_electron/mae_gradient_per_electron": MAEGradient(
                        mode="per electron"
                    ),
                    f"{split}_metrics/mae_initial_guess": MAEInitialGuess(mode="per molecule"),
                    f"{split}_metrics_per_electron/mae_initial_guess_per_electron": MAEInitialGuess(
                        mode="per electron"
                    ),
                }
            )

        self.train_metrics = create_metric_module_dict("train")
        self.val_metrics = create_metric_module_dict("val")
        self.test_metrics = create_metric_module_dict("test")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for each validation metric, also track the best value so far
        self.val_metrics_best = torch.nn.ModuleDict(
            {
                f"val_metrics_best/{key.split('/')[-1]}": MinMetric()
                for key in self.val_metrics.keys()
            }
        )

        # logging mixins
        self.logging_mixin_interval = logging_mixin_interval
        self.use_logging_mixins = self.logging_mixin_interval is not None
        self.val_logging_mixin_interval = (
            self.logging_mixin_interval // 5 if self.use_logging_mixins else None
        )
        if self.use_logging_mixins:
            self.logging_mixin_dict = locate_logging_mixins(self.net)
        else:
            self.logging_mixin_dict = {}
        self.show_logging_mixins_in_progress_bar = show_logging_mixins_in_progress_bar

        self.basis_info = basis_info

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        for metric in self.val_metrics.values():
            metric.reset()
        for metric_best in self.val_metrics_best.values():
            metric_best.reset()

    def activate_logging_mixins(self) -> None:
        """Activates the logging mixins."""
        for module in self.logging_mixin_dict.values():
            module.activate_logging()

    def deactivate_logging_mixins(self) -> None:
        """Deactivates the logging mixins."""
        for module in self.logging_mixin_dict.values():
            module.deactivate_logging()

    def forward(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Applies the forward pass of the model to the batch, and computes the energy gradients
        via backprop in the variational case (otherwise they are calculated directly).

        Args:
            batch (Batch): Batch object containing the data

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing (in order) the loss, the
            predicted energy, the predicted gradients and the predicted differences
        """
        if not self.variational:
            # model predicts energy, gradient and initial guess directly
            batch.coeffs.requires_grad = True  # for atomref image logging
            pred_energy, pred_gradients, pred_diff = self.net(batch)
        else:
            # enable gradient calculation, even when we use torch.no_grad at inference time
            with torch.enable_grad():
                # set the flag for the gradients to be calculated
                batch.coeffs.requires_grad = True
                # calculate the forward pass
                pred_energy, pred_diff = self.net(batch)

                # calculate the gradients from the predicted energy with autograd
                # Only at train time we retain the graph to calculate the gradients of the gradient
                pred_gradients = torch.autograd.grad(
                    pred_energy.sum(), batch.coeffs, create_graph=self.net.training
                )[0]

        return pred_energy, pred_gradients, pred_diff

    def sample_forward(self, sample: OFData) -> OFData:
        """Applies the forward pass of the model to the batch.

        Args:
            sample: OFData object.

        Returns:
            OFData: The batch with the model predictions added
        """
        pred_energy, pred_gradients, pred_diff = self.forward(sample)
        sample.add_item("pred_energy", pred_energy, Representation.SCALAR)
        sample.add_item("pred_gradient", pred_gradients, Representation.GRADIENT)
        sample.add_item("pred_diff", pred_diff, Representation.VECTOR)
        return sample

    @property
    def tensorboard_logger(self):
        """Get the tensorboard logger from the trainer."""
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_logger = logger.experiment
        return tb_logger

    def training_step(self, batch: Batch) -> dict:
        """Performs a single training step on a batch of data from the training set.

        Args:
            batch (Batch): A batch of data

        Returns:
            dict: A dict containing the loss, the model predictions, and the projected gradient differences.
        """
        compute_metrics_this_step = self.trainer.global_step % self.hparams.metric_interval == 0
        use_logging_mixins_this_step = (
            self.use_logging_mixins and self.trainer.global_step % self.logging_mixin_interval == 0
        )

        # activate logging mixins if steps matches logging_mixin_interval
        if use_logging_mixins_this_step:
            self.activate_logging_mixins()

        pred_energy, pred_gradients, pred_diff = self.forward(batch)

        # calculate and log the losses
        projected_gradient_difference = project_gradient_difference(pred_gradients, batch)
        weight_dict, loss_dict = self.loss_function(
            batch,
            pred_energy=pred_energy,
            projected_gradient_difference=projected_gradient_difference,
            pred_diff=pred_diff,
            pred_gradients=pred_gradients,
        )
        loss = sum([weight_dict[key] * loss_dict[key] for key in loss_dict.keys()])
        self.log(
            "train_loss/total",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=batch.batch_size,
        )
        self.train_loss(loss)
        log_dict = {f"train_loss/{key}": loss_dict[key] for key in loss_dict.keys()}
        self.log_dict(
            log_dict, on_step=True, on_epoch=False, prog_bar=True, batch_size=batch.batch_size
        )

        # update and log every metric
        if compute_metrics_this_step:
            for metric in self.train_metrics.values():
                metric(batch, pred_energy, projected_gradient_difference, pred_diff)
            self.log_dict(self.train_metrics, on_step=True, on_epoch=False, prog_bar=True)

        if use_logging_mixins_this_step:
            mixin_log_dict = {
                f"z_train_{prefix}/{key}": value
                for prefix, module in self.logging_mixin_dict.items()
                for key, value in module.log_dict.items()
            }
            tb_logger = self.tensorboard_logger
            if tb_logger is not None:
                for key, value in mixin_log_dict.items():
                    tb_logger.add_scalar(key, value, global_step=self.trainer.global_step)
            # For some reason, the below would significantly slow down training, even for iterations where it is not
            # called.
            # I think it's because lightning does not support logging at times other than on_step and on_epoch, i.e.
            # if you call log(..., on_step=True) once, the values will be logged at every step
            # Hence, we log directly to tensorboard (see above)
            # self.log_dict(
            #     mixin_log_dict,
            #     on_step=True,
            #     prog_bar=self.show_logging_mixins_in_progress_bar,
            #     batch_size=batch.batch_size
            # )
            self.deactivate_logging_mixins()
        return dict(
            loss=loss,
            model_outputs=dict(
                pred_energy=pred_energy,
                pred_gradients=pred_gradients,
                pred_diff=pred_diff,
            ),
            projected_gradient_difference=projected_gradient_difference,
        )

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        # One shouldn't call the metric.compute here if the metric is logged during the train step
        # self.console_logger.info(f"Train loss: {self.train_loss.compute()}")

    def on_validation_epoch_start(self) -> None:
        """Lightning hook that is called when a validation epoch starts."""
        super().on_validation_epoch_start()
        self.activate_logging_mixins()  # log values from logging mixins at every step during validation

    def validation_step(self, batch: Batch, batch_idx: int) -> dict:
        """Performs a single validation step on a batch of data from the validation set.

        Args:
            batch (Batch): A batch of data
            batch_idx (int): The index of the batch

        Returns:
            dict: A dict containing the loss, the model predictions, and the projected gradient differences.
        """
        compute_metrics_this_step = batch_idx % self.hparams.metric_interval == 0
        use_logging_mixins_this_step = (
            self.use_logging_mixins
            and self.trainer.global_step % self.val_logging_mixin_interval == 0
        )

        pred_energy, pred_gradients, pred_diff = self.forward(batch)
        # calculate the loss
        projected_gradient_difference = project_gradient_difference(pred_gradients, batch)
        weight_dict, loss_dict = self.loss_function(
            batch,
            pred_energy=pred_energy,
            projected_gradient_difference=projected_gradient_difference,
            pred_diff=pred_diff,
            pred_gradients=pred_gradients,
        )
        loss = sum([weight_dict[key] * loss_dict[key] for key in loss_dict.keys()])
        log_dict = {f"val_loss/{key}": loss_dict[key] for key in loss_dict.keys()}
        self.log_dict(
            log_dict, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.batch_size
        )
        # log losses
        self.val_loss(loss)
        self.log(
            "val_loss/total",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch.batch_size,
            prog_bar=True,
        )

        if use_logging_mixins_this_step:
            mixin_log_dict = {
                f"z_val_{prefix}/{key}": value
                for prefix, module in self.logging_mixin_dict.items()
                for key, value in module.log_dict.items()
            }
            tb_logger = self.tensorboard_logger
            if tb_logger is not None:
                for key, value in mixin_log_dict.items():
                    tb_logger.add_scalar(key, value, global_step=self.trainer.global_step)
        # update and log metrics
        if compute_metrics_this_step:
            for metric in self.val_metrics.values():
                metric(batch, pred_energy, projected_gradient_difference, pred_diff)
            self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)

        return dict(
            loss=loss,
            model_outputs=dict(
                pred_energy=pred_energy,
                pred_gradients=pred_gradients,
                pred_diff=pred_diff,
            ),
            projected_gradient_difference=projected_gradient_difference,
        )

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        for metric, (metric_best_key, metric_best) in zip(
            self.val_metrics.values(), self.val_metrics_best.items()
        ):
            scalar = metric.compute()  # get current val acc
            metric_best.update(scalar)  # update best so far val acc
            # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
            # otherwise metric would be reset by lightning after each epoch
            self.log(metric_best_key, metric_best.compute(), sync_dist=True, prog_bar=False)

        # deactivate logging mixins after validation
        self.deactivate_logging_mixins()

    def test_step(self, batch: Batch) -> None:
        """Performs a single test step on a batch of data from the test set.

        Args:
            batch (Batch): A batch of data
        """
        pred_energy, pred_gradients, pred_diff = self.forward(batch)
        # calculate the loss
        projected_gradient_difference = project_gradient_difference(pred_gradients, batch)
        weight_dict, loss_dict = self.loss_function(
            batch,
            pred_energy=pred_energy,
            projected_gradient_difference=projected_gradient_difference,
            pred_diff=pred_diff,
        )
        loss = sum([weight_dict[key] * loss_dict[key] for key in loss_dict.keys()])
        log_dict = {f"test_loss/{key}": loss_dict[key] for key in loss_dict.keys()}
        self.log_dict(log_dict, on_step=False, on_epoch=True, batch_size=batch.batch_size)
        self.test_loss(loss)

        self.log(
            "test_loss/total",
            self.test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.batch_size,
        )
        # update and log every metric
        for metric in self.test_metrics.values():
            metric(batch, pred_energy, projected_gradient_difference, pred_diff)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # One shouldn't call the metric.compute here if the metric is logged during the test step
        # self.console_logger.info(f"Test loss: {self.test_loss.compute()}")

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        Args:
            stage (str): Either `"fit"`, `"val"`, `"test"`, or `"predict"`.
        """
        # Doesn't work for double backward and doesn't provide a speed up currently.
        if self.hparams.compile:
            self.net = torch_geometric.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Returns:
            Dict[str, Any]: A dict containing the configured optimizers and learning-rate schedulers
                to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss/total",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
