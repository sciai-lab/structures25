from io import StringIO

import rich
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig
from rich.tree import Tree


def dict_to_tree(
    data: dict | DictConfig, tree: Tree = None, name: str = "Config", **tree_kwargs
) -> Tree:
    """Convert a nested dictionary to a Rich Tree."""
    if tree is None:
        tree = Tree(name, **tree_kwargs)

    for key, value in data.items():
        if isinstance(value, (dict, DictConfig)):
            dict_to_tree(value, tree.add(f"[cyan]{key}[/cyan]"))
        else:
            tree.add(f"[yellow]{key}[/yellow]: {value}")

    return tree


def tree_to_string(tree: Tree) -> str:
    """Convert a Rich Tree to a string."""
    s = StringIO()
    rich.print(tree, file=s)
    s.seek(0)
    tree_str = s.read()
    s.close()
    return tree_str


def log_config_text_to_tensorboard(cfg: DictConfig, trainer: Trainer) -> None:
    """Log the config tree to TensorBoard.

    Args:
        cfg: A DictConfig composed by Hydra.
        trainer: The Lightning trainer.
    """
    # send hparams to all loggers
    for logger in trainer.loggers:
        # for tensorboard logger, additionally log hparams as text
        if isinstance(logger, TensorBoardLogger):
            hparams_text = tree_to_string(dict_to_tree(cfg))
            hparams_text = "```\n" + hparams_text + "\n```"
            logger.experiment.add_text("hparams", hparams_text)
