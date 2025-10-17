"""Helper functions for configuring learning rate scheduling using Hydra."""

from typing import Callable, List

from torch.optim.lr_scheduler import ChainedScheduler
from torch.optim.optimizer import Optimizer


def chain_schedulers(optimizer: Optimizer, schedulers: List[Callable]) -> ChainedScheduler:
    """
    Chain multiple schedulers together, using :class:`torch.optim.lr_scheduler.ChainScheduler`.
    The point of this wrapper function is to make it easier to use in hydra config files:
    The optimizer has to be passed only once, in the same way as for a single scheduler.

    See ``configs/ml/model/schedulers/warmup_linear.yaml`` for an example.

    Args:
        optimizer: The optimizer to be scheduled.
        schedulers: A list of partially initialized schedulers, mapping optimizers to
            :class:`~torch.optim.lr_scheduler.LRScheduler`.

    Returns:
        A chained scheduler.

    """
    return ChainedScheduler([scheduler(optimizer) for scheduler in schedulers])
