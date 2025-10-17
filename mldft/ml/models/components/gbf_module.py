"""Compute edge features by applying Gaussian basis functions to the distances between nodes."""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.of_data import OFData


class GBFModule(nn.Module):
    """Module to calculate the edge attributes based on gaussian basis functions and the distance
    between nodes."""

    def __init__(self, num_gaussians: int = 10, normalized: bool = False) -> None:
        """Initializes the class. You can specify the number of gaussians and if the gaussians
        should be normalized. This function initializes the shift and scale parameters of the
        gaussians as learnable parameters.

        Args:
            num_gaussians (int, optional): Number of gaussian functions on which the edge features are calculated.
                Defaults to 10.
            normalized (bool, optional): Defines if the gaussians should be normalized. Defaults to False.
        """
        super().__init__()

        # use linspace to create the shifts of the gaussians in the range of 0 to 3
        # we use 3 as an initialization because the bond length of c-c is 3 Bohr
        self.shift = nn.Parameter(torch.linspace(0, 3, num_gaussians))

        # initialization of the scale parameter as 1
        self.scale = nn.Parameter(torch.ones(num_gaussians))

        self.normalized = normalized

    def forward(self, sample: OFData) -> Tensor:
        r"""Calculate the forward pass of the module. This function calculates the edge attributes
        as follows:

        .. math::
            e_{ij}^{k} = \exp\left(-\frac{1}{2} \left(\frac{\|r_i - r_j\| - \mu^k}
            {\sigma^k} \right)^2\right).

        A normalisation factor is added if the normalized flag is set to true.

        Args:
            sample (OFData): The input data containing the positions and edge_index.

        Returns:
            Tensor: The edge attributes. Shape: (E, num_gauss)
        """
        pos = sample.pos
        edge_index = sample.edge_index
        # get the position of the atom i and j from the edge index
        # i is the first row of the edge index and j is the second row
        pos_i = pos.index_select(0, edge_index[0])
        pos_j = pos.index_select(0, edge_index[1])

        # calculate the distance vector between the atoms
        dist = pos_i - pos_j

        # calculate the norm of the distance vector
        length = torch.norm(dist, dim=-1, keepdim=True)  # Shape E x 1

        squared_diff = torch.square(length - self.shift)
        squared_scale = torch.square(self.scale)

        # calculate the gaussian
        gaussian = torch.exp(-squared_diff / (2 * squared_scale))

        # if the gaussians should be normalized, divide by the normalization factor
        if self.normalized:
            gaussian = gaussian / (np.sqrt(2 * np.pi) * self.scale)
        return gaussian


@torch.jit.script
def gaussian(x, mean, std):
    """Gaussian kernel."""
    a = (2 * np.pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (std * a)


@torch.jit.script
def rbf(x, mean, std):
    """Radial basis function kernel, a Gaussian without the normalization."""
    return torch.exp(-0.5 * (((x - mean) / std) ** 2))


class GaussianLayer(nn.Module):
    def __init__(
        self,
        basis_info: BasisInfo,
        num_gaussians: int = 128,
        normalized: bool = True,
        directed=True,
        init_radius_range=(0, 3),
    ) -> None:
        """Initialize the GaussianLayer.

        Args:
            basis_info (BasisInfo): Used to determine the number of atom types.
            num_gaussians (int, optional): Number of gaussians to use. Defaults to 128.
            normalized (bool, optional): Whether to normalize the gaussians. Defaults to True.
            directed (bool, optional): Whether the learned means and stds are directed, i.e. whether a C-H edge gets
                the same embedding as the reverse H-C edge. Defaults to True.
            init_radius_range (Tuple[float, float], optional): Range for the initialization of the means.
                Defaults to (0, 3).
        """
        super().__init__()
        self.num_gaussians = (
            num_gaussians  # Different to previous implementation: sample both means and stds
        )
        # Different to previous implementation: Sample both means and stds
        self.means = nn.Embedding(1, num_gaussians)
        self.stds = nn.Embedding(1, num_gaussians)
        self.normalized = normalized
        self.directed = directed
        self.basis_info = basis_info

        self.n_types = basis_info.n_types
        self.n_edge_types = (
            basis_info.n_types**2
            if directed
            else basis_info.n_types * (basis_info.n_types + 1) // 2
        )

        self.mul = nn.Embedding(self.n_edge_types, 1)
        self.bias = nn.Embedding(self.n_edge_types, 1)

        nn.init.uniform_(self.means.weight, *init_radius_range)
        nn.init.uniform_(self.stds.weight, *init_radius_range)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, sample: OFData) -> Tensor:
        """Forward pass of the GaussianLayer."""
        pos = sample.pos
        edge_index = sample.edge_index

        # get the position of the atom i and j from the edge index
        # i is the first row of the edge index and j is the second row
        pos_i = pos.index_select(0, edge_index[0])
        pos_j = pos.index_select(0, edge_index[1])

        # same for atom types
        atom_i = sample.atom_ind.index_select(0, edge_index[0])
        atom_j = sample.atom_ind.index_select(0, edge_index[1])
        if self.directed:
            edge_types = atom_i * self.n_types + atom_j
        else:
            # effectively, we sort the atom types to make the edge type symmetric
            edge_types = torch.min(atom_i, atom_j) * self.n_types + torch.max(atom_i, atom_j)

        # calculate the edge lengths
        x = torch.norm(pos_i - pos_j, dim=-1, keepdim=True)
        length = x.clone()

        mul = self.mul(edge_types)
        bias = self.bias(edge_types)
        x = mul * x + bias
        x = x.expand(-1, self.num_gaussians)
        mean = self.means.weight.view(-1)
        std = self.stds.weight.view(-1).abs() + 1e-5

        if self.normalized:
            return gaussian(x.float(), mean, std).type_as(self.means.weight), length
        else:
            return rbf(x.float(), mean, std).type_as(self.means.weight), length
