# This file includes code from PyTorch Geometric (https://github.com/pyg-team/pytorch_geometric), licensed under the MIT License.
# The file was adapted to include the `list_keys` parameter.

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Self,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch
import torch.utils.data
from torch import Tensor
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch, Dataset
from torch_geometric.data.collate import _batch_and_ptr, _collate, repeat_interleave
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.data.storage import NodeStorage
from torch_geometric.typing import TensorFrame, torch_frame
from torch_geometric.utils import cumsum

T = TypeVar("T")
SliceDictType = Dict[str, Union[Tensor, Dict[str, Tensor]]]
IncDictType = Dict[str, Union[Tensor, Dict[str, Tensor]]]


class OFCollater:
    """Collater for OF-DFT data.

    Copy paste from torch_geometric.data.collate.Collater except for using
    our custom OFBatch and the `list_keys` attribute.
    """

    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        list_keys: Optional[Sequence[str]] = None,
    ):
        """Collater for OF-DFT data."""
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.list_keys = list_keys

    def __call__(self, batch: List[Any]) -> Any:
        """Collates a list of data objects into a single object."""
        elem = batch[0]
        if isinstance(elem, BaseData):
            # Changed to use our custom OFBatch
            return OFBatch.from_data_list(
                batch,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
                list_keys=self.list_keys,
            )
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, TensorFrame):
            return torch_frame.cat(batch, dim=0)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")


class OFBatch(Batch):
    """A batch object for OFData.

    Copied from `torch_geometric.data.Batch` with the addition
    of the `list_keys` attribute, which just appends attributes to a list instead of collating them, which
    is needed for square matrices of varying size.
    """

    @classmethod
    def from_data_list(
        cls,
        data_list: List[BaseData],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        list_keys: Optional[Sequence[str]] = None,  # Added to the original implementation
    ) -> Self:
        r"""Constructs a :class:`~torch_geometric.data.Batch` object from a list of
        :class:`~torch_geometric.data.Data` or :class:`~torch_geometric.data.HeteroData` objects.

        The assignment vector :obj:`batch` is created on the fly.
        In addition, creates assignment vectors for each key in
        :obj:`follow_batch`.
        Will exclude any keys given in :obj:`exclude_keys`.
        Keys in :obj:`list_keys` will be appended to a list instead
        of collated (useful for square matrices of varying size).
        """
        batch, slice_dict, inc_dict = collate(
            cls,
            data_list=data_list,
            increment=True,
            add_batch=not isinstance(data_list[0], Batch),
            follow_batch=follow_batch,
            exclude_keys=exclude_keys,
            list_keys=list_keys,
        )

        batch._num_graphs = len(data_list)  # type: ignore
        batch._slice_dict = slice_dict  # type: ignore
        batch._inc_dict = inc_dict  # type: ignore

        return batch


def collate(
    cls: Type[T],
    data_list: List[BaseData],
    increment: bool = True,
    add_batch: bool = True,
    follow_batch: Optional[Iterable[str]] = None,
    exclude_keys: Optional[Iterable[str]] = None,
    list_keys: Optional[Sequence[str]] = None,  # Added to the original implementation
) -> Tuple[T, SliceDictType, IncDictType]:
    """Collates a list of `data` objects into a single object of type `cls`.

    `collate` can handle both homogeneous and heterogeneous data objects by
    individually collating all their stores.
    In addition, `collate` can handle nested data structures such as
    dictionaries and lists.
    """

    if not isinstance(data_list, (list, tuple)):
        # Materialize `data_list` to keep the `_parent` weakref alive.
        data_list = list(data_list)

    if cls != data_list[0].__class__:  # Dynamic inheritance.
        out = cls(_base_cls=data_list[0].__class__)  # type: ignore
    else:
        out = cls()

    # Create empty stores:
    out.stores_as(data_list[0])  # type: ignore

    follow_batch = set(follow_batch or [])
    exclude_keys = set(exclude_keys or [])
    list_keys = set(list_keys or [])

    # Group all storage objects of every data object in the `data_list` by key,
    # i.e. `key_to_stores = { key: [store_1, store_2, ...], ... }`:
    key_to_stores = defaultdict(list)
    for data in data_list:
        for store in data.stores:
            key_to_stores[store._key].append(store)

    # With this, we iterate over each list of storage objects and recursively
    # collate all its attributes into a unified representation:

    # We maintain two additional dictionaries:
    # * `slice_dict` stores a compressed index representation of each attribute
    #    and is needed to re-construct individual elements from mini-batches.
    # * `inc_dict` stores how individual elements need to be incremented, e.g.,
    #   `edge_index` is incremented by the cumulated sum of previous elements.
    #   We also need to make use of `inc_dict` when re-constructuing individual
    #   elements as attributes that got incremented need to be decremented
    #   while separating to obtain original values.
    device: Optional[torch.device] = None
    slice_dict: SliceDictType = {}
    inc_dict: IncDictType = {}
    for out_store in out.stores:  # type: ignore
        key = out_store._key
        stores = key_to_stores[key]
        for attr in stores[0].keys():
            if attr in exclude_keys:  # Do not include top-level attribute.
                continue

            values = [store[attr] for store in stores]

            ###########################################################
            # This is the only change from the original implementation:
            if attr in list_keys:
                out_store[attr] = values
                if isinstance(values[0], Tensor):
                    if device is None:
                        device = values[0].device
                slice_dict[attr] = torch.arange(len(values) + 1, device=device)
                inc_dict[attr] = torch.zeros(len(values), device=device)
                continue
            ###########################################################

            # The `num_nodes` attribute needs special treatment, as we need to
            # sum their values up instead of merging them to a list:
            if attr == "num_nodes":
                out_store._num_nodes = values
                out_store.num_nodes = sum(values)
                continue

            # Skip batching of `ptr` vectors for now:
            if attr == "ptr":
                continue

            # Collate attributes into a unified representation:
            value, slices, incs = _collate(attr, values, data_list, stores, increment)

            # If parts of the data are already on GPU, make sure that auxiliary
            # data like `batch` or `ptr` are also created on GPU:
            if isinstance(value, Tensor) and value.is_cuda:
                device = value.device

            out_store[attr] = value

            if key is not None:  # Heterogeneous:
                store_slice_dict = slice_dict.get(key, {})
                assert isinstance(store_slice_dict, dict)
                store_slice_dict[attr] = slices
                slice_dict[key] = store_slice_dict

                store_inc_dict = inc_dict.get(key, {})
                assert isinstance(store_inc_dict, dict)
                store_inc_dict[attr] = incs
                inc_dict[key] = store_inc_dict
            else:  # Homogeneous:
                slice_dict[attr] = slices
                inc_dict[attr] = incs

            # Add an additional batch vector for the given attribute:
            if attr in follow_batch:
                batch, ptr = _batch_and_ptr(slices, device)
                out_store[f"{attr}_batch"] = batch
                out_store[f"{attr}_ptr"] = ptr

        # In case of node-level storages, we add a top-level batch vector it:
        if add_batch and isinstance(stores[0], NodeStorage) and stores[0].can_infer_num_nodes:
            repeats = [store.num_nodes or 0 for store in stores]
            out_store.batch = repeat_interleave(repeats, device=device)
            out_store.ptr = cumsum(torch.tensor(repeats, device=device))

    return out, slice_dict, inc_dict
