"""Toy architecture for testing purposes."""

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing, global_add_pool, knn_graph
from torchvision.ops import MLP

from mldft.ml.data.components.basis_info import BasisInfo


class ToyNet(nn.Module):
    """Toy net module for testing purposes."""

    def __init__(
        self,
        basis_info: BasisInfo,
        k_neighbors: int,
        hidden_dimension_energy: tuple[int, ...] = (128, 128, 128),
        hidden_dimension_coeffs: tuple[int, ...] = (128, 128, 128),
    ) -> None:
        """Initializes the ToyNet module.

        Args:
            basis_info (BasisInfo): BasisInfo object containing information about the basis for the dataset
            k_neighbors (int): Number of neighbors to consider in the graph creation
            hidden_dimension_energy (list[int], optional): list of dimensions of the hidden layer in the mlp calculating the energy. Defaults to [128,128,128].
            hidden_dimension_coeffs (list[int], optional): list of dimensions of the hidden layer in the mlp calculating the coefficients. Defaults to [128,128,128].
        """
        super().__init__()

        # calculate the maximum number of basis functions in an atom
        max_number_basis = basis_info.basis_dim_per_atom.max()

        # setup everything
        self.max_number_basis = max_number_basis
        self.k_neighbors = k_neighbors
        self.ToyInitialGuessDeltaModel = ToyInitialGuessDelta(
            max_number_basis, list(hidden_dimension_coeffs)
        )
        self.ToyEnergyModel = ToyEnergy(max_number_basis, list(hidden_dimension_energy))

    def forward(self, batch) -> tuple[Tensor, Tensor]:
        """Calculates the forward pass of the ToyNet module. By independently calculating the
        energy and the predicted delta to the ground state, to be used for the learned initial
        guess in OF-DFT.

        Args:
            batch (Batch): Batch object containing the data

        Returns:
            tuple[Tensor, Tensor]: The predicted energy and the predicted coefficients (in order)
        """

        # calculate the edge_index as a knn graph based on the positions of the nodes, and add self loops
        edge_index = knn_graph(x=batch.pos, k=self.k_neighbors, batch=batch.batch, loop=True)

        # split the coeffs by atom
        # this is needed to be able to use the message passing framework of pytorch geometric
        coeffs_list = list(batch.split_field_by_atom(batch.coeffs))

        # zero pad the coefficients
        for i in range(len(coeffs_list)):
            coeffs_list[i] = torch.nn.functional.pad(
                coeffs_list[i], (0, self.max_number_basis - coeffs_list[i].shape[0])
            )

        padded_coeffs = torch.stack(coeffs_list, dim=0)

        # calculate the forward pass of the energy and the coefficient differences
        initial_guess_delta = self.ToyInitialGuessDeltaModel.forward(
            batch, edge_index, padded_coeffs
        )
        energy = self.ToyEnergyModel.forward(batch, edge_index, padded_coeffs)

        return energy, initial_guess_delta


class ToyInitialGuessDelta(MessagePassing):
    """MessagePassing Module for calculating the inital_guess_delta."""

    def __init__(self, max_number_basis: int, hidden_dimensions: list[int]) -> None:
        """Initializes the ToyInitialGuessDelta module.

        Args:
            max_number_basis (int): maximal number of basis functions per atom
            hidden_dimensions (list[int]): dimensions of the hidden layers in the mlp
        """
        super().__init__()

        self.max_number_basis = max_number_basis

        # append the last layer to the hidden layers
        hidden_dimensions.append(max_number_basis)
        # need to add 3 for the positions
        self.mlp = MLP(max_number_basis + 3, hidden_dimensions)

    def forward(self, batch, edge_index: Tensor, padded_coeffs: Tensor) -> Tensor:
        """Calculates the forward pass of the ToyCoeffs module.

        Args:
            batch (Batch): Batch object containing the data
            edge_index (Tensor): edge index of the graph
            padded_coeffs (Tensor): zero padded coefficients

        Returns:
            Tensor: Coefficient inital_guess_delta
        """
        # Propagate the messages through the graph
        output = self.propagate(edge_index, pos=batch.pos, coeffs=padded_coeffs)

        # slice the tensor to the size of the coeffs (remove what was added by the zero padding)
        tmp_list = []
        for i in range(batch.pos.shape[0]):
            node_coeffs = output[i, : batch.n_basis_per_atom[i].int()]
            tmp_list.append(node_coeffs)
        # and concatenate the list of tensors to a single tensor
        out = torch.cat(tmp_list, dim=0)

        return out

    def message(self, pos_i: Tensor, pos_j: Tensor, coeffs_j: Tensor) -> Tensor:
        """Message function to calculate the message from node j to i.

        Args:
            pos_i (Tensor): position of the receiving node
            pos_j (Tensor): position of the sending node
            coeffs_j (Tensor): coefficients of the sending node

        Returns:
            Tensor: Message
        """

        # concatenate the difference of positions and the coefficients
        concatenated = torch.cat((pos_j - pos_i, coeffs_j), dim=-1)
        # and pass it through the mlp
        message = self.mlp(concatenated)

        return message


class ToyEnergy(MessagePassing):
    """Toy energy module for testing purposes."""

    def __init__(self, max_number_basis: int, hidden_dimensions: list[int]) -> None:
        """Initializes the ToyEnergy module.

        Args:
            max_number_basis (int): Maximal number of basis per atom
            hidden_dimensions (list[int]): number of dimensions of the hidden layers in the mlp
        """
        super().__init__()

        self.max_number_basis = max_number_basis
        # append the last layer onto the hidden layers
        hidden_dimensions.append(1)
        # need to add 3 for the positions
        self.mlp = MLP(max_number_basis + 3, hidden_dimensions)

    def forward(self, batch, edge_index: Tensor, padded_coeffs: Tensor) -> Tensor:
        """Calculates the forward pass of the ToyEnergy module.

        Args:
            batch (Batch): Batch object containing the data
            edge_index (Tensor): edge index of the graph
            padded_coeffs (Tensor): zero padded coefficients

        Returns:
            Tensor: predicted energy
        """

        # propagate the messages through the graph
        output = self.propagate(edge_index, pos=batch.pos, coeffs=padded_coeffs)

        # for each molecule in the batch, sum over all nodes to get the energy
        pooled_output = global_add_pool(output, batch.batch)

        # squeeze the last dimension because of how the labels are defined in the dataset
        return pooled_output.squeeze(dim=-1)

    def message(self, pos_i: Tensor, pos_j: Tensor, coeffs_j: Tensor) -> Tensor:
        """Message function to calculate the message from node j to i.

        Args:
            pos_i (Tensor): position of the receiving node
            pos_j (Tensor): position of the sending node
            coeffs_j (Tensor): coefficients of the sending node

        Returns:
            Tensor: Message
        """

        # concatenate the difference of positions and the coefficients
        concatenated = torch.cat((pos_j - pos_i, coeffs_j), dim=-1)
        # pass it through the mlp
        message = self.mlp(concatenated)

        return message
