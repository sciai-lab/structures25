# This file includes code from PyTorch Geometric (https://github.com/pyg-team/pytorch_geometric), licensed under the MIT License.
# The file was adapted to include the `list_keys` parameter.
"""DataLoader class for machine learning."""
from typing import List, Optional, Sequence, Union

from torch.utils.data import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter

from mldft.ml.data.components.of_batch import OFCollater


class OFLoader(DataLoader):
    """Data loader for OF-DFT data.

    Thin wrapper around torch_geometric.loader.DataLoader, setting "follow_batch" to ["coeffs"] by
    default, to facilitate splitting of all basis-function wise fields by molecule, e.g. "coeffs",
    "ground_state_coeffs", "gradient_label".
    """

    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = ("coeffs", "atomic_numbers"),
        exclude_keys: Optional[List[str]] = None,
        list_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """Data loader for OF-DFT data."""
        # Remove for PyTorch Lightning:
        kwargs.pop("collate_fn", None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=OFCollater(dataset, follow_batch, exclude_keys, list_keys),
            **kwargs,
        )
