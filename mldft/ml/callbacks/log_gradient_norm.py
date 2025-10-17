from typing import Dict, Union

import lightning as pl
import torch
from lightning import Callback
from lightning.pytorch.utilities import grad_norm
from loguru import logger
from torch.nn import Module
from torch.optim import Optimizer


class LogGradientNorm(Callback):
    """Log total and per-parameter gradient norm at every training step."""

    # maybe a good idea for the future; for now, just log every iteration

    # def __init__(self):
    #     super().__init__()
    # self.total_grad_norm_history = []
    # self.last_detailed_logging_step = None

    # def log_everything_this_step(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer: Optimizer):
    #     """Whether all individual gradient norms should be logged this step. Returns true in the first step, and if
    #     the total gradient norm is at least 10% larger than the running maximum over the last 100 steps,
    #     or this was the case in the previous step, or 10 steps ago.
    #     """
    #     if len(self.total_grad_norm_history) == 1:
    #         return True
    #     if trainer.global_step - self.last_detailed_logging_step in [1, 10]:
    #         return True
    #     if self.total_grad_norm_history[-1] > 1.1 * max(self.total_grad_norm_history[-100:-1]):
    #         return True
    #     return False

    def on_before_optimizer_step(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer: Optimizer
    ) -> None:
        """Log the gradients before the optimizer step."""

        with torch.no_grad():
            norm_dict = grad_norm(pl_module, norm_type=2)

        # self.total_grad_norm_history.append(norm_dict["grad_2.0_norm_total"].item())

        tb_logger = pl_module.tensorboard_logger
        if tb_logger is None:
            logger.warning("No tensorboard logger found. Skipping logging of gradient norms.")
            return

        # log total gradient norm in every step
        tb_logger.add_scalar(
            "grad_2.0_norm_total",
            norm_dict["grad_2.0_norm_total"],
            global_step=trainer.global_step,
        )

        if True:  # self.log_everything_this_step(trainer, pl_module, optimizer):
            norm_dict.pop("grad_2.0_norm_total")  # logged this already above
            for key, value in norm_dict.items():
                tb_logger.add_scalar(key, value, global_step=trainer.global_step)
            # self.last_detailed_logging_step = trainer.global_step


def parameter_norm(
    module: Module,
    norm_type: Union[float, int, str],
    group_separator: str = "/",
    learnable_only=True,
) -> Dict[str, float]:
    """Compute each parameter's norm and their overall norm.

    The overall norm is computed over all parameters together, as if they
    were concatenated into a single vector.

    Based on :class:`lightning.pytorch.utilities.grad_norm`.

    Args:
        module: :class:`torch.nn.Module` to inspect.
        norm_type: The type of the used p-norm, cast to float if necessary.
            Can be ``'inf'`` for infinity norm.
        group_separator: The separator string used by the logger to group
            the parameter norms in their own subfolder instead of the logs one.
        learnable_only: Whether to only consider parameters that have a gradient.

    Return:
        norms: The dictionary of p-norms of each parameter and
            a special entry for the total p-norm of the parameters viewed
            as a single vector.
    """
    with torch.no_grad():
        norm_type = float(norm_type)
        if norm_type <= 0:
            raise ValueError(
                f"`norm_type` must be a positive number or 'inf' (infinity norm). Got {norm_type}"
            )

        norms = {
            f"parameter_{norm_type}_norm{group_separator}{name}": p.data.norm(norm_type)
            for name, p in module.named_parameters()
            if ((p.grad is not None) if learnable_only else True)
        }
        if norms:
            total_norm = torch.tensor(list(norms.values())).norm(norm_type)
            norms[f"parameter_{norm_type}_norm_total"] = total_norm
    return norms


class LogParameterNorm(Callback):
    """Log total and per-parameter norm at every training step."""

    def on_before_optimizer_step(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer: Optimizer
    ) -> None:
        """Log the parameter norms before the optimizer step."""

        tb_logger = pl_module.tensorboard_logger
        if tb_logger is None:
            logger.warning("No tensorboard logger found. Skipping logging of parameter norms.")
            return

        parameter_norm_dict = parameter_norm(pl_module, norm_type=2)
        for key, value in parameter_norm_dict.items():
            tb_logger.add_scalar(key, value, global_step=trainer.global_step)
