import pytest
import torch

from mldft.ml.data.components.loader import OFLoader
from mldft.ml.models.components.toy_net import ToyNet
from mldft.utils.utils import set_default_torch_dtype


@set_default_torch_dtype(torch.float64)
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("k_neighbors", [3, 5])
def test_toy_net(dummy_basis_info, dummy_dataset_torch, batch_size, k_neighbors):
    """Tests if the shape of the toy net output is correct."""

    dataloader = OFLoader(dummy_dataset_torch, batch_size=batch_size, shuffle=False, num_workers=0)

    batch = next(iter(dataloader))

    net = ToyNet(dummy_basis_info, k_neighbors)

    out = net.forward(batch)

    # check if the shapes are right
    assert out[0].shape == batch.energy_label.shape
    assert out[1].shape == batch.gradient_label.shape
