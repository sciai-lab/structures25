"""Module for embedding density coefficients in an 'atom-hot' tensor.

The atom-hot embedding embeds the density coefficients for each atom in a molecule (which differ in
size depending on the atom type) in a tensor of fixed size. This is achieved in a way similar to a
one-hot embedding, where the resulting tensor consists of blocks, one for each atom type in the
basis and with size corresponding to the number of basis functions for this atom type. The
coefficients for a certain atom are then embedded in the corresponding block, while setting the
values in all other blocks to zero.

See figure 7 in [M-OFDFT]_ for a schematic representation of the embedding.
"""

import torch
from torch import Tensor

from mldft.ml.data.components.basis_info import BasisInfo


class AtomHotEmbedding(torch.nn.Module):
    """Embed coefficients in an 'atom-hot' tensor."""

    def __init__(self, embed_dim: int):
        """Initialize the AtomHotEmbedding module.

        Args:
            embed_dim (int): The embedding dimension is the same as the basis_info.n_basis
                attribute. We differentiate the terms here, to not confuse it with the
                OFData.n_basis attribute.
        """
        super().__init__()

        self.embed_dim = embed_dim

    @classmethod
    def from_basis_info(cls, basis_info: BasisInfo):
        """Initialize the AtomHotEmbedding module from a BasisInfo object."""
        return cls(basis_info.n_basis)

    def forward(
        self,
        coeffs: Tensor,
        basis_function_ind: Tensor,
        n_basis_per_atom: Tensor,
        coeff_ind_to_node_ind: Tensor,
    ) -> Tensor:
        """Embeds a tensor of coefficients concatenated over atoms in an 'atom-hot' tensor.

        Arguments:
            coeffs (Tensor): Tensor of density coefficients concatenated over atoms.
            basis_function_ind (Tensor[int]): Array holding the OFData's basis function indices.
                Will be used to embed coefficients in an atom-hot way.
            n_basis_per_atom (Tensor[int]): Tensor with number of basis functions per atom.
            coeff_ind_to_node_ind (Tensor[int]): Tensor mapping coefficient indices to
                node (=atom) indices.

        Returns:
            a_hot_coeffs (Tensor): Tensor of embedded density coefficients, with shape
            (n_atoms, self.embed_dim).
        """

        # initialize atom_hot embedded coeffs on same device and with same dtype as coeffs
        a_hot_coeffs = coeffs.new_zeros(size=(len(n_basis_per_atom), self.embed_dim))
        a_hot_coeffs[coeff_ind_to_node_ind, basis_function_ind] = coeffs

        return a_hot_coeffs
