import numpy as np
import torch
from numpy import ndarray
from opt_einsum import contract
from torch import Tensor


def _assert_shapes(coeffs: ndarray | Tensor, ao: ndarray | Tensor) -> None:
    """Assert that coefficient vector and atomic orbitals have the correct shapes.

    Args:
        coeffs: Coefficient vector, sha[e (n_ao).
        ao: Atomic orbitals on the grid.
            Shape: (n_grid, n_ao) or (n_deriv, n_grid, n_ao).
    """
    assert ao.ndim in (2, 3), (
        f"AO array is expected to have 2 or 3 dimensions " f"but has shape {ao.shape}"
    )
    assert coeffs.ndim == 1, (
        f"Coeffs array is expected to have 1 dimension " f"but has shape {coeffs.shape}"
    )
    assert coeffs.shape[0] == ao.shape[-1], (
        f"Number of coefficients ({coeffs.shape[0]}) does not match number of "
        f"atomic orbitals ({ao.shape[-1]})"
    )


def is_numpy(tensor: ndarray | Tensor) -> bool:
    """Check if tensor is a numpy array."""
    return isinstance(tensor, ndarray)


def is_torch(tensor: ndarray | Tensor) -> bool:
    """Check if tensor is a torch tensor."""
    return isinstance(tensor, Tensor)


def add_new_axis(tensor: ndarray | Tensor, axis=0) -> ndarray | Tensor:
    """Add a new axis to the tensor.

    Args:
        tensor: Input tensor.
        axis: Axis to add.

    Returns:
        tensor: Tensor with new axis added.
    """
    if is_numpy(tensor):
        return np.expand_dims(tensor, axis)
    elif is_torch(tensor):
        return tensor.unsqueeze(axis)
    else:
        raise TypeError("Input must be either a numpy array or a torch tensor")


def coeffs_to_rho(coeffs: ndarray | Tensor, ao: ndarray | Tensor) -> ndarray | Tensor:
    """Compute the electron density on the grid, using coeffs in the linear ansatz and (already
    evaluated) atomic orbitals on the grid.

    Args:
        coeffs: Coefficient vector.
        ao: Atomic orbitals on the grid.
            Shape: (n_grid, n_ao) or (n_deriv, n_ao, n_grid).

    Returns:
        rho: Electron density on the grid.
    """
    _assert_shapes(coeffs, ao)

    if ao.ndim == 3:
        ao = ao[0]

    rho = contract("ni,i", ao, coeffs)
    return rho


def coeffs_to_grad_rho(coeffs: ndarray | Tensor, ao: ndarray | Tensor) -> ndarray | Tensor:
    """Compute the gradient of the electron density on the grid, using coeffs in the linear ansatz
    and (already evaluated) atomic orbitals.

    Args:
        coeffs: Coefficient vector.
        ao: Atomic orbitals on the grid.
            Shape: (n_grid, n_ao) or (n_deriv, n_grid, n_ao).

    Returns:
        grad_rho: Gradient of the electron density on the grid.
    """
    _assert_shapes(coeffs, ao)
    assert (
        ao.ndim == 3 and ao.shape[0] >= 4
    ), "AO array is expected to contain at least first order derivatives."

    grad_rho = contract("xni,i->xn", ao[1:4], coeffs)
    return grad_rho


def coeffs_to_laplace_rho(coeffs: ndarray | Tensor, ao: ndarray | Tensor) -> ndarray | Tensor:
    """Compute the laplacian of the electron density on the grid, using coeffs in the linear ansatz
    and (already evaluated) atomic orbitals.

    Args:
        coeffs: Coefficient vector.
        ao: Atomic orbitals on the grid.
            Shape: (n_grid, n_ao) or (n_deriv, n_grid, n_ao).

    Returns:
        grad_rho: Laplacian of the electron density on the grid.
    """
    _assert_shapes(coeffs, ao)
    assert (
        ao.ndim == 3 and ao.shape[0] >= 10
    ), "AO array is expected to contain at least second order derivatives."

    dxdx, dydy, dzdz = 4, 7, 9
    laplace_rho = contract("xni,i->n", ao[[dxdx, dydy, dzdz], ...], coeffs)
    return laplace_rho


def concatenate(arrays: ndarray | Tensor, axis: int = 0):
    """Concatenate arrays along the given axis."""
    if all(is_numpy(array) for array in arrays):
        return np.concatenate(arrays, axis)
    elif all(is_torch(array) for array in arrays):
        return torch.cat(arrays, dim=axis)
    else:
        raise ValueError("All arrays must be of the same type (either all numpy or all torch)")


def coeffs_to_rho_and_derivatives(
    coeffs: ndarray | Tensor, ao: ndarray | Tensor, max_derivative_order: int
) -> ndarray | Tensor:
    """Compute the electron density and its derivatives up to order max_derivative_order on the
    grid, using coeffs in the linear ansatz and (already evaluated) atomic orbitals.

    Args:
        coeffs: Coefficient vector.
        ao: Atomic orbitals on the grid.
            Shape: (n_ao, n_grid) or (n_deriv, n_ao, n_grid).
        max_derivative_order: Maximum derivative order to compute, if zero, only the electron
            density is computed. If one, the gradient is concatenated. If two, the laplacian is
            additionally concatenated.

    Returns:
        rho_and_derivatives: Electron density and its derivatives on the grid.
            Shape depends on max_derivative:
                - if 0, shape is (1, n_grid)
                - if 1, shape is (4, n_grid)
                - if 2, shape is (5, n_grid)
    """
    if max_derivative_order < 0 or max_derivative_order > 2:
        raise ValueError(f"Invalid max_derivative_order {max_derivative_order}")

    results = [add_new_axis(coeffs_to_rho(coeffs, ao), axis=0)]
    if max_derivative_order >= 1:
        results.append(coeffs_to_grad_rho(coeffs, ao))
    if max_derivative_order >= 2:
        results.append(add_new_axis(coeffs_to_laplace_rho(coeffs, ao), axis=0))

    return concatenate(results)
