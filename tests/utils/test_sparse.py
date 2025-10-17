import numpy as np
import pytest
import torch

from mldft.utils.sparse import construct_block_diag_coo_indices_and_shape


@pytest.mark.parametrize(
    "block_shapes, expected_indices, expected_shape",
    [
        [((2, 2),), torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]]), (2, 2)],
        [
            ((2, 2), (3, 3)),
            torch.tensor(
                [[0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], [0, 1, 0, 1, 2, 3, 4, 2, 3, 4, 2, 3, 4]]
            ),
            (5, 5),
        ],
        [
            ((2, 1), (3, 2)),
            torch.tensor([[0, 1, 2, 2, 3, 3, 4, 4], [0, 0, 1, 2, 1, 2, 1, 2]]),
            (5, 3),
        ],
    ],
)
def test_construct_block_diag_coo_indices_and_shape_cases(
    block_shapes, expected_indices, expected_shape
):
    """Test the function `construct_block_diag_coo_indices_and_shape` for some cases."""
    indices, shape = construct_block_diag_coo_indices_and_shape(*block_shapes)
    assert torch.allclose(indices, expected_indices)
    assert shape == expected_shape


@pytest.mark.parametrize("block_shapes", [((2, 2),), ((2, 2), (3, 3)), ((2, 1), (3, 2))])
def test_construct_block_diag_coo_indices_and_shape_todense(block_shapes):
    """Test the function `construct_block_diag_coo_indices_and_shape` by constructing a block
    diagonal matrix and converting it to dense format."""
    # create tensors for the blocks
    tensors = [
        torch.arange(np.prod(block_shape)).reshape(block_shape) for block_shape in block_shapes
    ]

    expected_dense = torch.block_diag(*tensors)

    indices, shape = construct_block_diag_coo_indices_and_shape(*block_shapes)
    sparse = torch.sparse_coo_tensor(
        indices, torch.cat([b.flatten() for b in tensors]), size=shape
    )
    dense = sparse.to_dense()

    assert torch.allclose(dense, expected_dense)
