"""Implements :class:`LoggingMixin`, which can be used easily log from any :class:`torch.nn.Module`

Example:

    >>> class MyLayer(torch.nn.Module, LoggingMixin):
    >>>     def forward(self, x, y):
    >>>         z = x + y
    >>>         # easily log intermediate value
    >>>         self.log(x=x, z_intermediate=z)
    >>>         z = z ** 2
    >>>         self.log(z_final=z)
    >>>         return z
"""

from numbers import Number

import numpy as np
import torch
from torch import Tensor


class LoggingMixin:
    """A mixin class for logging.

    Use in your ``nn.Module`` to log values during training and validation.
    """

    def activate_logging(self) -> None:
        """Activate logging."""
        self._logging_active = True

    def deactivate_logging(self) -> None:
        """Deactivate logging."""
        self._logging_active = False
        # clear log_dict
        self._log_dict = dict()

    @property
    def log_dict(self) -> dict:
        """A dictionary of values to log."""
        if not hasattr(self, "_log_dict"):
            self._log_dict = dict()
        return self._log_dict

    def _process_log_dict(self, log_dict: dict) -> dict:
        """process a dict of things to be logged: log scalars directly, otherwise compute mean, std, abs_max"""

        processed_log_dict = {}
        for key, value in log_dict.items():
            if not isinstance(value, (Number, np.ndarray, Tensor)):
                raise ValueError(
                    "Currently, only the logging of numbers, arrays and tensors is implemented"
                )
            if isinstance(value, Number):
                processed_log_dict[key] = value
                continue
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if value.numel() == 1:
                processed_log_dict[key] = value.item()
                continue
            # value is a tensor with multiple elements
            value = value.detach()
            processed_log_dict[f"mean_{key}"] = value.mean().item()
            processed_log_dict[f"std_{key}"] = value.std().item()
            processed_log_dict[f"abs-max_{key}"] = value.abs().max().item()
        return processed_log_dict

    def log(self, **log_dict: dict) -> None:
        """Log values, if logging was activated."""

        if not getattr(self, "_logging_active", False):
            return

        self.log_dict.update(self._process_log_dict(log_dict))


def _locate_logging_mixins_generator(module: torch.nn.Module, prefix: str = ""):
    """Recursively locate all LoggingMixin instances in a module and its children."""
    if isinstance(module, LoggingMixin):
        yield prefix, module
    for name, child in module.named_children():
        yield from _locate_logging_mixins_generator(child, prefix + "." + name if prefix else name)


def locate_logging_mixins(module: torch.nn.Module) -> dict:
    """Locate all LoggingMixin instances in a module and its (named) children.

    Args:
        module: The module to search.

    Returns:
        A dictionary mapping the dotted 'paths' within the module to the LoggingMixin instances.
    """
    return dict(_locate_logging_mixins_generator(module))
