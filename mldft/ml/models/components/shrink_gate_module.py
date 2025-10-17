"""Shrink gate."""

import torch
from torch import nn


class ShrinkGateModule(torch.nn.Module):
    """Module class for the shrink gate."""

    def __init__(self, lambda_co: float = 10.0, lambda_mul: float = 0.02) -> None:
        """Initializes the two parameters needed for the shrink gate.

        Args:
            lambda_co (float, optional): Scaling parameter of output. Defaults to 10.0.
            lambda_mul (float, optional): Scaling parameter inside tanh. Defaults to 0.02.
        """
        super().__init__()
        self.lambda_co = torch.nn.Parameter(torch.Tensor([lambda_co]))
        self.lambda_mul = torch.nn.Parameter(torch.Tensor([lambda_mul]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculates the forward pass of the module. Where the components of the x Tensor are
        transformed as follows: x -> lambda_co * tanh(lambda_mul * x) (component-wise)

        Args:
            x (Tensor): Input of the module.

        Returns:
            Tensor: The transformed tensor.
        """

        return self.lambda_co * torch.tanh(self.lambda_mul * x)


class PerBasisFuncShrinkGateModule(torch.nn.Module):
    """Module class for the shrink gate."""

    def __init__(self, embed_dim: int, lambda_co: float = 10.0, lambda_mul: float = 0.02) -> None:
        """Initializes the two parameters needed for the shrink gate.

        Args:
            embed_dim (int): Number of existing basis functions same as the basis_info.n_basis
                or sum(basis_dim_per_atom).
            lambda_co (float, optional): Scaling parameter of output. Defaults to 10.0.
            lambda_mul (float, optional): Scaling parameter inside tanh. Defaults to 0.02.
        """
        super().__init__()
        self.inner_factor = nn.Parameter(torch.ones(embed_dim) * lambda_mul)
        self.outer_factor = nn.Parameter(torch.ones(embed_dim) * lambda_co)

    def forward(self, coeffs: torch.Tensor):
        """Calculates the forward pass of the module. Where the components of the x Tensor are
        transformed as follows: x_ij -> lambda_co_j * tanh(lambda_mul_j * x_ij) where i is the
        index of the atom in the batch, j is the basis function index.

        Args:
            coeffs (Tensor): Atom-hot embedded coefficients of shape (n_atoms, embed_dim)

        Returns:
            Tensor: The shrunk tensor of shape (n_atoms, embed_dim)
        """
        return torch.tanh(coeffs * self.inner_factor.reshape(1, -1)) * self.outer_factor.reshape(
            1, -1
        )
