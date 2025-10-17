import pytest
import torch

from mldft.ml.models.components.g3d_layer import G3DLayer

torch.manual_seed(1337)


@pytest.fixture
def g3d_layer_setup():
    """Setup for the G3DLayer tests."""
    in_channels = 16
    heads = 4
    edge_dim = 1
    g3d_layer_setup = G3DLayer(in_channels, heads, edge_dim)

    x = torch.randn((10, in_channels))  # 10 nodes with 'in_channels' features each
    edge_index = torch.randint(0, 10, (2, 20))  # Randomly generated edges between nodes
    edge_attr = torch.randn((20, edge_dim))  # 20 edges with 'edge_dim' features each

    return g3d_layer_setup, x, edge_index, edge_attr


def test_forward(g3d_layer_setup):
    """Test the forward pass of the G3DLayer.

    Args:
        g3d_layer_setup: The setup for the G3DLayer tests.
    """
    g3d_layer, x, edge_index, edge_attr = g3d_layer_setup
    output = g3d_layer(x, edge_index, torch.zeros(10, dtype=torch.long), edge_attr)
    assert output.shape == (10, g3d_layer.in_channels), "Output shape mismatch."


def test_propagate(g3d_layer_setup):
    """Test the propagate method of the G3DLayer.

    Args:
        g3d_layer_setup: The setup for the G3DLayer tests.
    """
    g3d_layer, x, edge_index, edge_attr = g3d_layer_setup
    query_i = x.view(-1, g3d_layer.heads, g3d_layer.in_channels // g3d_layer.heads)
    key_j = x.view(-1, g3d_layer.heads, g3d_layer.in_channels // g3d_layer.heads)
    value_j = x.view(-1, g3d_layer.heads, g3d_layer.in_channels // g3d_layer.heads)
    edge_attr = edge_attr
    output = g3d_layer.propagate(
        edge_index,
        query=query_i,
        key=key_j,
        value=value_j,
        edge_attr=edge_attr,
        size=None,
        length=None,
    )
    assert output.shape == (
        10,
        g3d_layer.heads,
        g3d_layer.in_channels // g3d_layer.heads,
    ), "Output shape mismatch."


def test_reset_parameters(g3d_layer_setup):
    """Test the reset_parameters method of the G3DLayer.

    Args:
        g3d_layer_setup: The setup for the G3DLayer tests.
    """
    g3d_layer, _, _, _ = g3d_layer_setup
    original_key_weight = g3d_layer.linear_in.weight.data.clone()
    g3d_layer.reset_parameters()
    assert not torch.equal(
        original_key_weight, g3d_layer.linear_in.weight.data
    ), "Weights should have changed after reset."
