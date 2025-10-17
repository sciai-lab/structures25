"""Wrapper for einsums intended to supply the fastest einsum implementation and perform the
required type conversions."""
import numpy as np
import torch


def einsum(einsum_notation: str, *tensors: np.ndarray | torch.Tensor) -> np.ndarray:
    """Einsum wrapper that accepts numpy arrays and torch tensors.

    It is a wrapper intended to use the in our opinion faster implementation of einsum and perform
    the required type conversions. Currently, the pytorch einsum implementation seems to be faster
    than numpy even on the cpu and using numpy optimization.

    Args:
        einsum_notation: einsum notation string
        *tensors: tensors to be multiplied

    Returns:
        einsum result

    Note:
        this function might be prone to changes in the future
    """
    tensors = [torch.from_numpy(t) if isinstance(t, np.ndarray) else t for t in tensors]
    # maybe at some point we want to circumvent the back and forth conversion
    # though they are relatively fast (at least on CPUs ...)
    return torch.einsum(einsum_notation, *tensors).numpy()
