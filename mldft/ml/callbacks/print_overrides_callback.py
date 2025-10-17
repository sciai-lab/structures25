import logging
from pathlib import Path

import lightning.pytorch as pl
import rich
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks.callback import Callback

from mldft.utils.environ import get_mldft_model_path
from mldft.utils.rich_utils import format_table_rich

log = logging.getLogger(__name__)


def _get_hydra_overrides():
    """Returns the hydra config overrides as a string."""
    hydra_config = HydraConfig.get()
    return hydra_config.overrides["task"]


class PrintOverrides(Callback):
    """Prints the hydra config overrides to the console."""

    def __init__(self, compact: bool = False):
        """
        Args:
            compact: If True, the overrides are printed in a compact form. Otherwise, they are printed in a table.
                Defaults to False.
        """
        super().__init__()
        self.compact = compact

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Prints the hydra config overrides to the console."""
        try:
            override_strings = _get_hydra_overrides()
        except ValueError:
            log.warning("Hydra config not found. Overrides will not be printed.")
            return

        if self.compact:
            log.info("Overrides: \n" + " ".join(override_strings))
        else:
            cols = [
                ("key", [s.split("=")[0] for s in override_strings]),
                ("value", ["removed" if "~" in s else s.split("=")[1] for s in override_strings]),
            ]
            if hasattr(trainer, "logger") and hasattr(trainer.logger, "log_dir"):
                path = Path(trainer.logger.log_dir)

                # print path relative to dft_model_path if possible
                dft_model_path = get_mldft_model_path()
                if path.is_relative_to(dft_model_path):
                    path = "$DFT_MODELS" / path.relative_to(dft_model_path)

                title = f'Overrides at "{path}":'
            else:
                title = "Overrides:"
            print(title)
            rich.print(format_table_rich(*cols))
