"""Callback mechanism for hydra  jobs.

Adapted from https://github.com/paquiteau/hydra-callbacks
"""
from __future__ import annotations

import logging
import os

from hydra.experimental.callback import Callback
from omegaconf import DictConfig, open_dict

callback_logger = logging.getLogger("hydra.callbacks")


def dummy_run(config: DictConfig, **kwargs: None) -> None:
    """Do nothing."""


class AnyRunCallback(Callback):
    """Abstract Callback that execute on any run."""

    def __init__(self, enabled: bool = True):
        callback_logger.debug("Init %s", self.__class__.__name__)

        self.enabled = enabled
        if not self.enabled:
            # don't do anything if not enabled
            self._on_anyrun_start = dummy_run  # type: ignore
            self._on_anyrun_end = dummy_run  # type: ignore

    def on_run_start(self, config: DictConfig, **kwargs: None) -> None:
        """Execute before a single run."""
        callback_logger.debug("run start callback %s", self.__class__.__name__)
        self._on_anyrun_start(config, **kwargs)

    def on_multirun_start(self, config: DictConfig, **kwargs: None) -> None:
        """Execute before a multi run."""
        callback_logger.debug("(multi)run start callback %s", self.__class__.__name__)
        self._on_anyrun_start(config, **kwargs)

    def _on_anyrun_start(self, config: DictConfig, **kwargs: None) -> None:
        """Execute before any run."""

    def on_run_end(self, config: DictConfig, **kwargs: None) -> None:
        """Execute before a single run."""
        callback_logger.debug("run end callback %s", self.__class__.__name__)
        self._on_anyrun_end(config, **kwargs)

    def on_multirun_end(self, config: DictConfig, **kwargs: None) -> None:
        """Execute before a multi run."""
        callback_logger.debug("(multi)run end callback %s", self.__class__.__name__)
        self._on_anyrun_end(config, **kwargs)

    def _on_anyrun_end(self, config: DictConfig, **kwargs: None) -> None:
        """Execute before any run."""


class GitInfo(AnyRunCallback):
    """Callback that check git infos and log them.

    Parameters
    ----------
    clean
        if True, will fail if the repo is not clean
    """

    def __init__(self, clean: bool = False):
        super().__init__()
        self.clean = clean

    def _on_anyrun_start(self, config: DictConfig, **kwargs: None) -> None:
        """Execute before any run."""
        try:
            import git
        except ImportError:
            callback_logger.error("GitPython is not installed, aborting")
            return

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        is_dirty = repo.is_dirty(untracked_files=True)
        branch_name = repo.active_branch.name  # Get the branch name

        callback_logger.warning(f"Git sha: {sha}, branch: {branch_name}, dirty: {is_dirty}")

        if is_dirty and self.clean:
            callback_logger.error("Repo is dirty, aborting")
            os._exit(1)

        # Add git info to config
        with open_dict(config):
            config.git = {"sha": sha, "branch": branch_name, "is_dirty": is_dirty}
