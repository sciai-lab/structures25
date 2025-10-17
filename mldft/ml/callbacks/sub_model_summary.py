from typing import Any, Dict, List, Tuple

import numpy as np
from lightning.pytorch.callbacks import RichModelSummary
from lightning.pytorch.utilities.model_summary import get_human_readable_count


class SubModelSummary(RichModelSummary):
    r"""Generates a summary of all layers in a submodule of a
    :class:`~lightning.pytorch.core.LightningModule`."""

    def __init__(
        self, max_depth: int = 1, path_in_model: str = None, **summarize_kwargs: Any
    ) -> None:
        """Generates a summary of all layers in a submodule of a
        :class:`~lightning.pytorch.core.LightningModule`.

        Args:
            max_depth: The maximum depth of layer nesting that the summary will include. A value of 0 turns the
                layer summary off.
            path_in_model: The path to the submodule of interest. Defaults to None, which will summarize the entire model.
            **summarize_kwargs: Additional arguments to pass to the `summarize` method.

        Example::

            >>> from lightning.pytorch import Trainer
            >>> from lightning.pytorch.callbacks import ModelSummary
            >>> trainer = Trainer(callbacks=[SubModelSummary(max_depth=1, path_in_model="net.submodule_of_interest")])

            This will summarize ``pl_module.net.submodule_of_interest``.
        """
        super().__init__(max_depth=max_depth, **summarize_kwargs)
        self.path_in_model = path_in_model

    def summarize(
        self,
        summary_data: List[Tuple[str, List[str]]],
        total_parameters: int,
        trainable_parameters: int,
        model_size: float,
        total_training_modes: Dict[str, int] = None,
        **summarize_kwargs: Any,
    ) -> None:
        """Adapted summarize method which filters the summary_data to only include rows in
        ``self.path_in_model``."""
        assert summary_data[1][0] == "Name"

        # filter summary data
        ind = [i for i, x in enumerate(summary_data[1][1]) if x.startswith(self.path_in_model)]
        filtered_summary_data = [(x[0], [x[1][i] for i in ind]) for x in summary_data]

        # count number of parameters outside the sub-model, and include in table if > 0
        num_other_parameters = int(
            np.sum([int(x[1][0]) for i, x in enumerate(summary_data) if i not in ind])
        )
        if num_other_parameters > 0:
            filtered_summary_data[0][1].append(str(len(filtered_summary_data[0][1])))
            filtered_summary_data[1][1].append(f"outside of {self.path_in_model}")
            filtered_summary_data[2][1].append("")
            filtered_summary_data[3][1].append(
                "{:<{}}".format(get_human_readable_count(num_other_parameters), 10)
            )

        if total_training_modes is None:
            # for backward compatibility with older versions of Lightning (<2.4)
            super().summarize(
                filtered_summary_data,
                total_parameters,
                trainable_parameters,
                model_size,
                **summarize_kwargs,
            )
        else:
            super().summarize(
                filtered_summary_data,
                total_parameters,
                trainable_parameters,
                model_size,
                total_training_modes,
                **summarize_kwargs,
            )
