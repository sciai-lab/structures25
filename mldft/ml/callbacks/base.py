from typing import Any

import pytorch_lightning as pl
from lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from mldft.ml.callbacks.timing import CallbackTiming, EveryIncreasingInterval
from mldft.ml.models.mldft_module import MLDFTLitModule


def default_log_timing():
    """Returns a default log timing, which logs with exponentially increasing intervals."""
    return EveryIncreasingInterval()


class OnStepCallbackWithTiming(Callback):
    """Base class for timed callbacks, which execute at certain intervals.

    They execute on step, during training and validation, at independently specified timings.
    """

    def __init__(
        self,
        train_timing: CallbackTiming = None,
        val_timing: CallbackTiming = None,
        name: str = "auto",
    ) -> None:
        """Initializes the callback. The name is used for logging.

        Args:
            train_timing: The :class:`CallbackTiming` object specifying how often the callback is to be
                called during training.
            val_timing: The :class:`CallbackTiming` object specifying how often the callback is to be
                called during validation.
            name: The name of the callback. If ``"auto"``, the class name is used, minus "Log" if the
                class name starts with that.
        """
        super().__init__()
        self.val_log_timing = val_timing if val_timing is not None else default_log_timing()
        self.train_log_timing = train_timing if train_timing is not None else default_log_timing()
        if name == "auto":
            self.name = self.__class__.__name__
            if self.name.startswith("Log"):
                self.name = self.name[3:]
        else:
            self.name = name

    def execute(
        self,
        trainer: pl.Trainer,
        pl_module: MLDFTLitModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        split: str,
    ) -> None:
        """Executes the callback, e.g. logs a figure to tensorboard.

        Args:
            pl_module: The lightning module.
            trainer: The lightning trainer.
            outputs: The outputs of the lightning module.
            batch: The batch data.
            split: The split, either ``"train"`` or ``"val"``.
        """
        raise NotImplementedError

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: MLDFTLitModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Executes the callback via :meth:`execute`, if the timing matches."""
        if not self.train_log_timing.call_now(trainer.global_step):
            return

        self.execute(
            pl_module=pl_module,
            trainer=trainer,
            outputs=outputs,
            batch=batch,
            split="train",
        )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: MLDFTLitModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        **kwargs,
    ) -> None:
        """Executes the callback via :meth:`execute`, if the timing matches.

        Can happen only for the first validation batch.
        """
        # only log on the first validation batch, as tensorboard does not show multiple images for the same step anyway
        if batch_idx != 0:
            return
        if not self.val_log_timing.call_now(trainer.global_step):
            return

        self.execute(
            pl_module=pl_module,
            trainer=trainer,
            outputs=outputs,
            batch=batch,
            split="val",
        )
