"""The NodeEmbedding module initializes the hidden node features :math:`h`.

It combines three parts: A simple embedding layer assigns learnable weights to each of the
atomic numbers :math:`Z`. Density coefficients :math:`\\tilde{p}` are embedded in an 'atom-hot' way,
mapping all coefficients to the same dimension. A further processing of a ShrinkGateModule maps
the coefficients into a bounded space, facilitating a stable optimization. Afterwards, an MLP
projects these features to the hidden node dimension. To encode the chemical environment, the
pairwise distances encoded by Gaussian-Basis-Functions are aggregated and passed through an MLP.
The three parts are then summed to yield the hidden node features :math:`h`, ready to serve as
input for the following stack of G3D-layers.
"""

from __future__ import annotations

import torch
from numpy import ndarray
from torch import Tensor
from torch.nn import Embedding, Module, SiLU
from torch_geometric.utils import scatter

from mldft.ml.models.components.density_coeff_embedding import AtomHotEmbedding
from mldft.ml.models.components.mlp import MLP
from mldft.ml.models.components.shrink_gate_module import (
    PerBasisFuncShrinkGateModule,
    ShrinkGateModule,
)
from mldft.utils.log_utils.logging_mixin import LoggingMixin


def smooth_falloff(
    value: torch.Tensor, falloff_end: float, falloff_start: float = 0.0
) -> torch.Tensor:
    """Calculates a smooth falloff value using a cosine function.

    The function returns a tensor with values in the range [0, 1] such that:
      - For values less than or equal to `falloff_start`, the function returns 1.
      - For values greater than or equal to `falloff_end`, the function returns 0.
      - For values between `falloff_start` and `falloff_end`, it returns a smoothly
        interpolated value between 1 and 0 using a cosine function.

    The cosine interpolation is computed as:
        0.5 * (cos(pi * (value - falloff_start) / (falloff_end - falloff_start)) + 1)

    Args:
      value: The input tensor.
      falloff_end: The value at which the falloff reaches 0.
      falloff_start: The value at which the falloff begins (default is 0).

    Returns:
      A tensor representing the smooth falloff value, between 0 and 1.
    """
    if falloff_start >= falloff_end:
        raise ValueError("falloff_start must be less than falloff_end")

    # Compute the interpolation factor for values in the decay range.
    fraction = (value - falloff_start) / (falloff_end - falloff_start)
    cosine_falloff = 0.5 * (torch.cos(torch.pi * fraction) + 1)

    # Use torch.where to assign:
    # - 1 for values <= falloff_start,
    # - 0 for values >= falloff_end, and
    # - cosine_falloff for values in between.
    result = torch.where(
        value <= falloff_start, 1, torch.where(value >= falloff_end, 0, cosine_falloff)
    )

    return result


class NodeEmbedding(Module, LoggingMixin):
    """The NodeEmbedding module creates node features from density, atomic number and distance
    embeddings."""

    def __init__(
        self,
        n_atoms: int,
        basis_dim_per_atom: Tensor | ndarray,
        basis_atomic_numbers: Tensor | ndarray,
        atomic_number_to_atom_index: Tensor | ndarray,
        out_channels: int,
        dst_in_channels: int,
        p_hidden_channels: int = 32,
        p_num_layers: int = 3,
        p_activation: callable = SiLU,
        p_dropout: float = 0.0,
        dst_hidden_channels: int = 32,
        dst_num_layers: int = 3,
        dst_activation: callable = SiLU,
        dst_dropout: float = 0.0,
        lambda_co: float = None,
        lambda_mul: float = None,
        use_per_basis_func_shrink_gate: bool = False,
        cutoff: float | None = None,
        cutoff_start: float = 0.0,
    ) -> None:
        r"""Initialize the NodeEmbedding module.

        Args:
            n_atoms (int): Number of atom types in the dataset.
            basis_dim_per_atom (Tensor[int] | ndarray[int]): Basis Dimensions per atomic number.
            basis_atomic_numbers (Tensor[int] | ndarray[int]): Atomic numbers in the basis.
            atomic_number_to_atom_index (Tensor[int] | ndarray[int]): Mapping from atomic number
                to atom index, e.g. basis_dim_per_atom[atomic_number_to_atom_index[1]] yields the
                basis dimensions for atomic number 1.
            out_channels (int): Number of output channels for the hidden node representation :math:`h`.
            dst_in_channels (int): Number of input channels of GBF-transformed pairwise distances
                :math:`\mathcal{E}`.
            p_hidden_channels (int): Number of hidden channels for the MLP of density coefficients.
                Defaults to 32.
            p_num_layers (int): Number hidden layers for the MLP of density coefficients.
                Defaults to 3.
            p_activation (callable): Activation function for the MLP of density coefficients.
            dst_hidden_channels (int): Number of hidden channels for the MLP for the distances.
            dst_num_layers (int): Number hidden layers for the MLP of GBF-transformed distances.
                Defaults to 3.
            dst_activation (callable): Activation function for the MLP of GBF-transformed
            lambda_co (float): lambda_co parameter for the ShrinkGateModule.
            lambda_mul (float): lambda_mul parameter for the ShrinkGateModule.
            use_per_basis_func_shrink_gate (bool): Whether to use a per-basis-function shrink gate or not.
                Mainly to allow backwards compatibility for older checkpoints.
            cutoff: The cutoff radius for the distance embedding. If None, no cutoff is applied.
        """

        super().__init__()
        self.n_atoms = n_atoms  # Maybe only at forward pass?
        self.out_channels = out_channels
        self.basis_dim_per_atom = basis_dim_per_atom
        self.basis_atomic_numbers = basis_atomic_numbers
        self.atomic_number_to_atom_index = atomic_number_to_atom_index
        self.atom_hot_embedding = AtomHotEmbedding(sum(basis_dim_per_atom))
        self.dst_in_channels = dst_in_channels
        self.p_in_channels = sum(basis_dim_per_atom)
        self.z_embed = Embedding(n_atoms, out_channels)
        self.cutoff = cutoff
        self.cutoff_start = cutoff_start
        if use_per_basis_func_shrink_gate:
            self.shrink_gate = PerBasisFuncShrinkGateModule(
                sum(basis_dim_per_atom), lambda_co, lambda_mul
            )
        else:
            self.shrink_gate = ShrinkGateModule(lambda_co, lambda_mul)
        self.mlp_p = MLP(
            in_channels=self.p_in_channels,
            hidden_channels=[p_hidden_channels for _ in range(p_num_layers - 1)] + [out_channels],
            activation_layer=p_activation,
            dropout=p_dropout,
        )
        self.mlp_dst = MLP(
            in_channels=self.dst_in_channels,
            hidden_channels=[dst_hidden_channels for _ in range(dst_num_layers - 1)]
            + [out_channels],
            activation_layer=dst_activation,
            dropout=dst_dropout,
        )
        self.reset_parameters()

    @classmethod
    def from_basis_info(
        cls,
        basis_info,
        out_channels: int,
        dst_in_channels: int,
        p_hidden_channels: int = 32,
        p_num_layers: int = 3,
        p_activation: callable = SiLU,
        p_dropout: float = 0.0,
        dst_hidden_channels: int = 32,
        dst_num_layers: int = 3,
        dst_activation: callable = SiLU,
        dst_dropout: float = 0.0,
        lambda_co: float = None,
        lambda_mul: float = None,
        use_per_basis_func_shrink_gate: bool = False,
        cutoff: float | None = None,
        cutoff_start: float = 0.0,
    ) -> NodeEmbedding:
        """Initialize the NodeEmbedding module from a BasisInfo object.

        The arguments pertaining to the basis_info object, i.e. basis_dim_per_atom,
        basis_atomic_numbers and atomic_number_to_atom_index are extracted from the basis_info.
        For the remaining arguments, see :meth:`__init__` for details on other arguments.
        """

        return cls(
            n_atoms=basis_info.n_types,
            basis_atomic_numbers=basis_info.atomic_numbers,
            basis_dim_per_atom=basis_info.basis_dim_per_atom,
            atomic_number_to_atom_index=basis_info.atomic_number_to_atom_index,
            out_channels=out_channels,
            dst_in_channels=dst_in_channels,
            p_hidden_channels=p_hidden_channels,
            p_num_layers=p_num_layers,
            p_activation=p_activation,
            p_dropout=p_dropout,
            dst_hidden_channels=dst_hidden_channels,
            dst_num_layers=dst_num_layers,
            dst_activation=dst_activation,
            dst_dropout=dst_dropout,
            lambda_co=lambda_co,
            lambda_mul=lambda_mul,
            use_per_basis_func_shrink_gate=use_per_basis_func_shrink_gate,
            cutoff=cutoff,
            cutoff_start=cutoff_start,
        )

    def reset_parameters(self) -> None:
        """Reset all parameters of the NodeEmbedding module."""

        for layer in self.mlp_p:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for layer in self.mlp_dst:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

        self.z_embed.reset_parameters()

    def aggregate_distances(
        self, edge_attributes: Tensor, edge_index: Tensor, n_atoms: int
    ) -> Tensor:
        """Aggregate distances (edge features) for each atom over connected atoms.

        Args:
            edge_attributes (Tensor[float]): Edge features (GBF transformed distances) of shape
                (num_edges, dst_in_channels).
            edge_index (Tensor[int]): Edge indices of shape (2, num_edges).
            n_atoms: Number of atoms in the sample.

        Returns:
            aggregated_features (Tensor[float]): Aggregated edge features of
                shape (n_atoms, dst_in_channels).
        """

        source_nodes = edge_index[0]  # 'source' atoms

        aggregated_features = scatter(src=edge_attributes, index=source_nodes, dim_size=n_atoms)

        return aggregated_features

    def forward(
        self,
        coeffs: Tensor | ndarray,
        atom_ind: Tensor | ndarray,
        basis_function_ind: Tensor | ndarray,
        n_basis_dim_per_atom: Tensor | ndarray,
        coeff_ind_to_node_ind: Tensor,
        distance_embedding: Tensor,
        edge_index: Tensor,
        batch: Tensor = None,
        length: Tensor = None,
    ) -> Tensor:
        r"""Forward pass of the NodeEmbedding module.

        Passes the list of density coefficients :math:`\tilde{p}` through and embedding and the
        ShrinkGate, embeds the atomic_numbers :math:`Z` and aggregates the distances :math:`\mathcal{E}` over
        edges. After passing the density coefficients and distance features through MLPs, the
        node features :math:`h` are calculated as the sum of the three terms.

        Args:
            coeffs (Tensor[float]): Density coefficients :math:`\tilde{p}` of varying shape
                but length n_atoms.
            atom_ind (Tensor[int]): Atomic indices of shape (n_atoms, 1).
            basis_function_ind (Tensor[int]): Array holding the OFData's basis function indices.
                Will be used to embed coefficients in an atom-hot way.
            n_basis_dim_per_atom (Tensor[int]): Number of basis functions per atom.
            coeff_ind_to_node_ind (Tensor[int]): Tensor mapping coefficient indices to
                node (=atom) indices.
            distance_embedding (Tensor[float]): GBF transformed distances of shape
                (num_edges, dst_in_channels). dst_in_channels correspond to edge_channels.
            edge_index (Tensor[int]): Edge indices of shape (2, num_edges).
            batch (Tensor, optional): Batch tensor for LayerNorm inside the MLPs.
            length (Tensor, optional): Edge lengths for the cutoff function.

        Returns:
            h (Tensor[float]): Node features :math:`h` of shape (n_atoms, out_channels).
        """

        if isinstance(coeffs, ndarray):
            coeffs = torch.from_numpy(coeffs).float()
        if isinstance(atom_ind, ndarray):
            atom_ind = torch.from_numpy(atom_ind).int()
        if isinstance(basis_function_ind, ndarray):
            basis_function_ind = torch.from_numpy(basis_function_ind).int()

        p = self.atom_hot_embedding(
            coeffs=coeffs,
            basis_function_ind=basis_function_ind,
            n_basis_per_atom=n_basis_dim_per_atom,
            coeff_ind_to_node_ind=coeff_ind_to_node_ind,
        )
        p = self.shrink_gate(p)
        p = self.mlp_p(p, batch)

        z = self.z_embed(atom_ind.squeeze())

        if self.cutoff is not None:
            distance_embedding = distance_embedding * smooth_falloff(
                length, self.cutoff, self.cutoff_start
            )

        dst_sum = self.aggregate_distances(
            edge_attributes=distance_embedding,
            edge_index=edge_index,
            n_atoms=atom_ind.shape[0],
        )

        dst_sum = self.mlp_dst(dst_sum, batch)

        self.log(p=p, z=z, dst_sum=dst_sum)
        # self.log(p_std=p.std(), z_std=z.std(), dst_sum_std=dst_sum.std())

        h = p + z + dst_sum

        return h

    def __setstate__(self, state: dict) -> None:
        """This method is called during unpickling.

        If 'cutoff_start' is missing (as would be the case with an older checkpoint), it will be
        added with a default value.
        """
        # Update the state dictionary first
        self.__dict__.update(state)
        # If the new attribute is missing, add it with the default value.
        if not hasattr(self, "cutoff"):
            self.cutoff = None
        if not hasattr(self, "cutoff_start"):
            self.cutoff_start = 0.0
