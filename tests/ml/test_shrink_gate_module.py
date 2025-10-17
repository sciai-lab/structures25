import pytest
import torch
from torch import Tensor

from mldft.ml.models.components.shrink_gate_module import PerBasisFuncShrinkGateModule


@pytest.mark.parametrize(
    "x, expected",
    [
        (
            Tensor([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]),
            Tensor(
                [
                    -0.79829764,
                    -0.59928101,
                    -0.39978680,
                    -0.19997334,
                    0.00000000,
                    0.19997334,
                    0.39978680,
                    0.59928101,
                    0.79829764,
                ]
            ),
        ),
        (
            Tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]),
            Tensor(
                [
                    [0.19997334, 0.39978680, 0.59928101, 0.79829764],
                    [0.19997334, 0.39978680, 0.59928101, 0.79829764],
                    [0.19997334, 0.39978680, 0.59928101, 0.79829764],
                ]
            ),
        ),
    ],
)
def test_shrink_gate_module(x: Tensor, expected: Tensor) -> None:
    """Tests the ShrinkGateModule. Tests the forward pass and the gradient calculation.

    Args:
        x (Tensor): Tensor, which is fed forward through the shrink gate.
        expected (Tensor): Expected output.
    """
    module = PerBasisFuncShrinkGateModule(embed_dim=x.shape[-1])
    output = module.forward(x)
    assert torch.allclose(output, expected)
