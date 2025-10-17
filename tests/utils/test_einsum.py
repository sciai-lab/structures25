import numpy as np
import pytest
import torch

from mldft.utils.einsum import einsum


@pytest.mark.parametrize(
    "einsum_notation, tensors",
    [
        ("ij,jk", [np.random.rand(10, 10), np.random.rand(10, 10)]),
        ("ij,jk", [torch.rand(10, 10), torch.rand(10, 10)]),
        (
            "ij,ia,ah",
            [np.random.rand(10, 10), np.random.rand(10, 10), np.random.rand(10, 10)],
        ),
        ("ij,ia,ah", [torch.rand(10, 10), torch.rand(10, 10), torch.rand(10, 10)]),
    ],
)
def test_einsum(einsum_notation, tensors):
    expected = np.einsum(einsum_notation, *tensors, optimize=True)
    actual = einsum(einsum_notation, *tensors)

    assert np.allclose(expected, actual)
    assert isinstance(actual, np.ndarray)
