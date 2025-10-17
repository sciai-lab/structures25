"""Module for the projMINAO guess from [M-OFDFT]_."""
from typing import List

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import SiLU

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.models.components.mlp import MLP
from mldft.ml.preprocess.dataset_statistics import DatasetStatistics


class InitialGuessDeltaModule(nn.Module):
    """Module to calculate the initial guess delta based on the node features computed by previous
    layers."""

    def __init__(
        self,
        input_size: int,
        basis_info: BasisInfo,
        hidden_layers: List[int],
        activation_function: nn.Module | None = SiLU,
        dataset_statistics: DatasetStatistics | None = None,
        weigher_key: str = "initial_guess_only",
        dropout: float = 0.0,
    ) -> None:
        """Initializes the class. calculates the atom_ptr object, which is needed for reverting the
        atom hot encoding. Also initializes the mlp with the given parameters.

        Args:
            input_size (int): Input size of the mlp
            basis_info (BasisInfo): BasisInfo object
            activation_function (Module, optional): Activation function for the MLP. Defaults to SiLU.
            hidden_layers (Tuple[int, ...], optional): Configurations of the hidden layers in the MLP.
                Defaults to (32, 32, 32).
            dataset_statistics (DatasetStatistics, optional): :class:`DatasetStatistics` object. Defaults to None.
                If not None, this will be used to offset and scale the predicted delta according to per-coefficient
                mean and standard deviation as given in the :class:`DatasetStatistics` object.
            weigher_key (str): Selects the sample weigher for the dataset statistics.
            dropout (float, optional): Dropout probability of the MLP. Defaults to 0.0.
        """

        super().__init__()

        # calculate the atom_ptr object for splitting the atom hot encoding, saves the start index of each atom
        self.atom_ptr = np.zeros(len(basis_info.basis_dim_per_atom), dtype=np.int32)
        self.atom_ptr[1:] = np.cumsum(basis_info.basis_dim_per_atom)[:-1]

        self.number_of_basis_per_atom = basis_info.basis_dim_per_atom

        self.with_scale_shift = dataset_statistics is not None
        if self.with_scale_shift:
            self.register_buffer(
                "mean",
                torch.as_tensor(
                    dataset_statistics.load_statistic(weigher_key, "initial_guess_delta/mean")
                ),
            )
            self.register_buffer(
                "std",
                torch.as_tensor(
                    dataset_statistics.load_statistic(weigher_key, "initial_guess_delta/std")
                ),
            )

        output_size = basis_info.basis_dim_per_atom.sum().item()

        # Need to append the output layer to the hidden layers
        mlp_layers = hidden_layers + [output_size]

        self.mlp = MLP(
            in_channels=input_size,
            hidden_channels=mlp_layers,
            activation_layer=activation_function,
            dropout=dropout,
            disable_dropout_last_layer=True,
            disable_activation_last_layer=True,
        )

    def forward(
        self,
        x: Tensor,
        basis_function_ind: Tensor,
        coeff_ind_to_node_ind: Tensor,
        batch: Tensor = None,
    ) -> Tensor:
        """Calculates the forward pass of the module. The coefficients are passed through an MLP
        and then the atom-hot-encoding is reverted.

        Args:
            x (Tensor): Activations of the previous layer. Shape: (n_atoms, input_size).
            basis_function_ind (Tensor): Indices of the basis functions in the molecule. Shape: (n_atoms,).
            coeff_ind_to_node_ind (Tensor[int]): Tensor mapping coefficient indices to
                node (=atom) indices.
            batch (Tensor, optional): Batch tensor for LayerNorm inside the mlp.

        Returns:
            Tensor: the initial_guess_delta in one vector. Shape: (n_basis,)
        """

        out = self.mlp(x, batch)

        result = out[coeff_ind_to_node_ind, basis_function_ind]

        if self.with_scale_shift:
            result *= self.std[basis_function_ind]
            result += self.mean[basis_function_ind]

        return result
