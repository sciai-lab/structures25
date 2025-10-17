import os

from lightning.pytorch.callbacks import ModelCheckpoint
from typing_extensions import override


class ModelCheckpointWithPermissions(ModelCheckpoint):
    """Adapted callback from `lightning.callbacks.ModelCheckpoint` to set file permissions to 0o640
    when saving."""

    @override
    def _save_checkpoint(self, trainer, filepath: str) -> None:
        """Save checkpoint according to lightning code then set file permissions to 0o640."""
        super()._save_checkpoint(trainer, filepath)
        # Owner read write, group read
        os.chmod(filepath, 0o640)
