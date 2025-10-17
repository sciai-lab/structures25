from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from mldft.utils.log_utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def rich_to_str(rich_object: Table | Any) -> str:
    """Converts a Rich object to a string."""
    console = Console(file=StringIO(), width=120)
    console.print(rich_object)
    return console.file.getvalue()


def add_as_string_option(func) -> Callable:
    """Decorator that adds an option to a function that returns a Rich object to return a string
    instead."""

    def wrapper(*args, **kwargs):
        """Wrapper function."""
        if kwargs.pop("as_string", False):
            return rich_to_str(func(*args, **kwargs))
        return func(*args, **kwargs)

    return wrapper


@add_as_string_option
def format_table_rich(
    *cols: Tuple[str, List[str]],
    col_kwargs: List[Dict[str, Any]] = None,
    title=None,
) -> Table:
    """Formats a table using the Rich library.

    Args:
        *cols: A list of tuples of the form (title, values) or (title, values_section_1, values_section_2, ..).
        col_kwargs: A list of dictionaries with keyword arguments for each column.
        title: The title of the table.
        as_string: Whether to return the table as a string. Default is ``False``.

    Returns:
        A Rich Table object.
    """
    table = Table(title=title, style="dim", title_style="dim")
    if col_kwargs is None:
        for c in cols:
            table.add_column(c[0])
    else:
        for c, kw in zip(cols, col_kwargs):
            table.add_column(c[0], **kw)
    for sec in range(1, len(cols[0])):
        table.add_section()
        sec_lengths = [len(col[sec]) for col in cols]
        assert all(
            length == sec_lengths[0] for length in sec_lengths
        ), f"All sections must have the same length! Got {sec_lengths}."
        for i in range(len(cols[0][sec])):
            table.add_row(*[c[sec][i] for c in cols])
    return table


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
    print_to_console: bool = True,
    return_str: bool = False,
) -> None | str:
    """Prints the contents of a DictConfig as a tree structure using the Rich library.

    Args:
        cfg: A DictConfig composed by Hydra.
        print_order: Determines in what order config components are printed. Default is
        ``("data", "model", "callbacks", "logger", "trainer", "paths", "extras")``.
        resolve: Whether to resolve reference fields of DictConfig. Default is ``False``.
        save_to_file: Whether to export config to the hydra output folder. Default is ``False``.
        print_to_console: Whether to print config tree to console. Default is ``True``.
        return_str: Whether to return config tree as a string. Default is ``False``.

    Returns:
        None or str: If ``return_str=True``, returns config tree as a string.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else log.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    if print_to_console:
        rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)

    # return config tree as a string
    if return_str:
        s = StringIO()
        rich.print(tree, file=s)
        s.seek(0)
        config_tree_str = s.read()
        s.close()
        return config_tree_str


@rank_zero_only
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompts user to input tags from command line if no tags are provided in config.

    :param cfg: A DictConfig composed by Hydra.
    :param save_to_file: Whether to export tags to the hydra output folder. Default is ``False``.
    """
    if not cfg.get("tags"):
        if "id" in HydraConfig().cfg.hydra.job:
            raise ValueError("Specify tags before launching a multirun!")

        log.warning("No tags provided in config. Prompting user to input tags...")
        tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
        tags = [t.strip() for t in tags.split(",") if t != ""]

        with open_dict(cfg):
            cfg.tags = tags

        log.info(f"Tags: {cfg.tags}")

    if save_to_file:
        with open(Path(cfg.paths.output_dir, "tags.log"), "w") as file:
            rich.print(cfg.tags, file=file)
