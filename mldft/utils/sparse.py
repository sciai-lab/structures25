import torch


def construct_block_diag_coo_indices_and_shape(
    *block_shapes: tuple[int, int],
    device: torch.device = None,
) -> (torch.Tensor, tuple[int, int]):
    """Construct the indices for a block diagonal sparse matrix in COOrdinate format.

    >>> blocks = [torch.ones((2, 2)), torch.ones((3, 3))]
    >>> indices, shape = construct_block_diag_coo_indices_and_shape(*[b.shape for b in blocks])
    >>> torch.sparse_coo_tensor(indices, torch.cat([b.flatten() for b in blocks]), size=shape).to_dense()

    Args:
        block_shapes: The shapes of the blocks.
        device: Which device to use.

    Returns:
        torch.Tensor: The indices of the blocks.
        tuple: The shape of the block diagonal matrix.
    """
    block_offsets = torch.cumsum(torch.tensor([(0, 0)] + list(block_shapes), device=device), dim=0)
    shape = tuple(block_offsets[-1])

    indices = []
    for block_size, block_offset in zip(block_shapes, block_offsets):
        indices.append(
            torch.stack(
                torch.meshgrid(
                    torch.arange(block_size[0], device=device),
                    torch.arange(block_size[1], device=device),
                    indexing="ij",
                )
            ).reshape(2, -1)
            + block_offset[:, None]
        )
    indices = torch.cat(indices, dim=-1)

    return indices, shape


def construct_block_diag_coo_tensor_indices_and_shape_from_sparse(
    *blocks: torch.Tensor,
    device: torch.device = None,
) -> (torch.Tensor, tuple[int, int]):
    """Construct the indices for a block diagonal sparse matrix in COOrdinate format where the
    blocks are sparse tensors.

    Args:
        blocks: The blocks of the block diagonal tensor.
        device: Which device to use.
    Returns:
        torch.Tensor: The indices of the blocks.
        tuple: The shape of the block diagonal matrix.
    """

    block_shapes = torch.tensor([b.shape for b in blocks], dtype=torch.int, device=device)
    block_shapes = torch.cat((torch.zeros((1, 2), dtype=torch.int), block_shapes), dim=0)
    block_offsets = torch.cumsum(block_shapes, dim=0)
    block_indices = [b.coalesce().indices() for b in blocks]

    shape = tuple(block_offsets[-1])
    indices = torch.tensor([], dtype=torch.int, device=device)

    for idx, block_offset in zip(block_indices, block_offsets):
        indices = torch.cat((indices, idx + block_offset.unsqueeze(1)), dim=1)

    return indices, shape


def construct_block_diag_coo_tensor(*blocks: torch.Tensor) -> torch.Tensor:
    """Construct a block diagonal tensor from the given blocks.

    >>> blocks = [torch.ones((2, 2)), torch.ones((3, 3))]
    >>> construct_block_diag_tensor(*blocks)

    Args:
        blocks:         The blocks of the block diagonal tensor.

    Returns:
        torch.Tensor:   The block diagonal tensor.
    """

    assert not any(b.is_sparse for b in blocks) != all(
        b.is_sparse for b in blocks
    ), "Either all or none of the blocks must be sparse tensors."
    device = blocks[0].device
    if any(b.is_sparse for b in blocks):
        indices, shape = construct_block_diag_coo_tensor_indices_and_shape_from_sparse(
            *blocks, device=device
        )
        values = torch.cat([b.coalesce().values() for b in blocks])
    else:
        indices, shape = construct_block_diag_coo_indices_and_shape(
            *[b.shape for b in blocks], device=device
        )
        values = torch.cat([b.flatten() for b in blocks])

    return torch.sparse_coo_tensor(indices, values, size=shape)
