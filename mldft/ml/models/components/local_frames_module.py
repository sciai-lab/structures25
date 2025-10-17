"""Calculates the local frames of the atoms in the molecule."""

import numpy as np
import torch
from e3nn.o3 import Irreps
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing, knn_graph

from mldft.ml.data.components.of_data import OFData
from mldft.utils.local_frames import (
    local_frames_from_rel_positions,
    pyscf_to_e3nn_local_frames_matrix,
    transform_coeffs_to_local,
)


class LocalBasisModule(MessagePassing):
    """MessagePassing Module, which calculates the local basis on each node of the graph."""

    def __init__(
        self, ignore_hydrogen: bool = True, use_three_atoms_for_basis: bool = False
    ) -> None:
        """This class is a MessagePassing module which calculates the local basis of the atoms in
        the molecule when given positions on a graph.

        Args:
            ignore_hydrogen (bool, optional): If true, the two closest heavy atoms are used to construct the
                local bases. Defaults to True.
            use_three_atoms_for_basis (bool, optional): If true, the three closest atoms are used to construct the basis
        """
        super().__init__()

        self.ignore_hydrogen = ignore_hydrogen

        # distance where the dummy atoms are placed
        self.dummy_distance = 1e6

        # Number of neighbors to consider. (In this case fixed to 3)
        if use_three_atoms_for_basis:
            self.k = 3
        else:
            self.k = 2

    def aggregate(self, inputs: Tensor) -> Tensor:
        """Aggregates the messages from the neighboring atoms.

        Args:
            inputs (Tensor): The relative positions of the neighboring atoms. Shape: (n_edges=k*n_atom, 3)

        Returns:
            Tensor: The local frames of the atoms in the molecule.
        """
        # Reshape to [num_nodes, k, num_features]
        inputs = inputs.view(-1, self.k, 3)

        length = torch.norm(inputs, dim=-1)

        if self.k == 3:
            # Sort the inputs by the length of the vectors
            sorted_indices = torch.argsort(length, dim=-1)
            sorted_indices_inv = torch.argsort(sorted_indices, dim=-1)

            pos_1 = inputs[sorted_indices_inv == 0, :]
            pos_2 = inputs[sorted_indices_inv == 1, :]
            pos_3 = inputs[sorted_indices_inv == 2, :]

            local_basis = local_frames_from_rel_positions(pos_1, pos_2, pos_3)
        else:
            assert self.k == 2
            # Sort the inputs by the length of the vectors
            # use <= for the edge case of two neighbors with equal distance
            diff = length[:, 0] - length[:, 1]
            pos_1 = torch.where(diff[:, None] <= 0, inputs[:, 0], inputs[:, 1])
            pos_2 = torch.where(diff[:, None] <= 0, inputs[:, 1], inputs[:, 0])

            local_basis = local_frames_from_rel_positions(pos_1, pos_2)

        return local_basis

    def forward(
        self,
        pos: Tensor,
        atomic_numbers: Tensor | None = None,
        batch: Tensor | None = None,
    ) -> Tensor:
        """Calculates the forward pass of the module.

        Args:
            pos (Tensor): The positions of the atoms.
            atomic_numbers (Tensor): The atomic numbers of the atoms in the molecule
            batch (Tensor, optional): The pytorch geometric batch. Defaults to None.

        Returns:
            Tensor: The local frames of the atoms in the molecule.
        """

        if not self.ignore_hydrogen:
            # construct a simple knn graph
            # if there are not enough atoms, i.e., one more than the number of neighbors, add dummy atoms to the graph
            if len(pos) - 1 < self.k:
                n_dummy_atoms = self.k - len(pos) + 1
            else:
                n_dummy_atoms = 0

            assert (
                2 * max(torch.linalg.norm(pos, dim=-1)) < self.dummy_distance
            ), "The dummy distance might be too small for the molecule"

            # add dummy atoms to the graph at self.dummy_distance. If dummy_atoms is 0, this does nothing
            pos = torch.cat(
                [
                    pos,
                    self.dummy_distance
                    * torch.randn(
                        [n_dummy_atoms] + list(pos[0].size()),
                        dtype=pos.dtype,
                        layout=pos.layout,
                        device=pos.device,
                    ),
                ]
            )
            edge_index = knn_graph(pos, self.k, batch, loop=False, flow=self.flow)

        else:
            heavy_atom_mask = atomic_numbers != 1
            # if there are not enough heavy atoms, i.e., one more than the number of neighbors, add dummy atoms to the
            # graph
            if heavy_atom_mask.sum() - 1 < self.k:
                n_dummy_atoms = self.k - heavy_atom_mask.sum() + 1
            else:
                n_dummy_atoms = 0

            heavy_atom_mask = torch.cat(
                [
                    heavy_atom_mask,
                    torch.ones(
                        n_dummy_atoms,
                        dtype=heavy_atom_mask.dtype,
                        device=heavy_atom_mask.device,
                    ),
                ]
            )

            # construct a knn graph with the three closest heavy atoms for each atom otherwise add dummy atoms
            assert (
                atomic_numbers is not None
            ), "atom_ind must be provided if ignore_hydrogen is True"
            assert batch is None, "batching is not supported if ignore_hydrogen is True"

            heavy_atom_ind = torch.argwhere(heavy_atom_mask)
            pos = torch.cat(
                [
                    pos,
                    self.dummy_distance
                    * torch.randn(
                        [n_dummy_atoms] + list(pos[0].size()),
                        dtype=pos.dtype,
                        layout=pos.layout,
                        device=pos.device,
                    ),
                ]
            )

            # construct distance matrix between heavy atoms and all atoms
            dist_mat = torch.cdist(pos, pos[heavy_atom_mask])
            dist_order = heavy_atom_ind[torch.argsort(dist_mat, dim=1, descending=False)]

            assert torch.all(dist_order[heavy_atom_mask, 0].eq(heavy_atom_ind)), (
                f"the closest heavy atom is not itself: "
                f"{dist_order[heavy_atom_mask, 0]} != {heavy_atom_ind}"
            )

            edge_index_neighbors = torch.cat(
                [
                    # take the k closest heavy atoms for each hydrogen
                    dist_order[~heavy_atom_mask, : self.k].flatten(),
                    # take the second to k+1 closest heavy atoms for each heavy atom, as the closest one is itself
                    dist_order[heavy_atom_mask, 1 : self.k + 1].flatten(),
                ],
                dim=0,
            )
            edge_index_centers = torch.cat(
                [
                    # take the hydrogen atoms
                    torch.arange(pos.shape[0])[~heavy_atom_mask].repeat_interleave(self.k),
                    # take the heavy atoms
                    torch.arange(pos.shape[0])[heavy_atom_mask].repeat_interleave(self.k),
                ],
                dim=0,
            )
            edge_index = torch.stack([edge_index_neighbors, edge_index_centers], dim=0)

            # sort the edge index by the center atoms. this is necessary for the aggregation!
            edge_index = edge_index[:, torch.argsort(edge_index[1])]

        # assert that the edge index is sorted
        assert torch.all(edge_index[1, 1:] >= edge_index[1, :-1])

        # propagate the messages through the knn-graph
        result = self.propagate(edge_index, pos=pos)
        # remove the dummy atoms again
        if n_dummy_atoms > 0:
            # remove the dummy atoms from the result
            result = result[:-n_dummy_atoms, :, :]

        return result

    def message(self, pos_i, pos_j) -> Tensor:
        """Return the relative position of the neighboring atoms as a message.

        Args:
            pos_i (Tensor): The position of the atom, from the second row of the edge index.
            pos_j (Tensor): The position of the neighboring atom, from the first row of the edge index.

        Returns:
            Tensor: The relative position of the neighboring atom.
        """
        return pos_j - pos_i


class LocalFramesModule(nn.Module):
    """This Module calculates the transformed coefficients as a nn.Module.

    It splits the coefficients by atom and transforms them into the local frame individually.
    :class:`LocalFramesTransformMatrix` can be used to do it in parallel using sparse matrices.
    """

    def __init__(self) -> None:
        """Initializes the module."""
        super().__init__()
        self.local_basis_module = LocalBasisModule()

    def forward(
        self,
        coeffs: list[Tensor],
        irreps: list[Irreps],
        pos: Tensor,
        atomic_numbers: Tensor | None = None,
        batch: Tensor | None = None,
    ) -> list[Tensor]:
        """Calculates the forward pass of the module.

        Args:
            coeffs (list[Tensor]): coefficients of the atoms in the molecule.
            irreps (list[Irreps]): irreps of the atoms in the molecule.
            pos (Tensor): positions of the atoms in the molecule.
            atomic_numbers (Tensor): The atomic numbers of the atoms in the molecule
            batch (Tensor, optional): The pytorch geometric batch. Defaults to None.

        Returns:
            Tensor: The transformed coefficients into the local frame.
        """

        bases = self.local_basis_module.forward(pos, atomic_numbers, batch)
        for i in range(pos.size(0)):
            # calculate the transformed coefficients
            coeffs[i] = transform_coeffs_to_local(coeffs[i], irreps[i], bases[i])

        return coeffs


class LocalFramesTransformMatrixSparse(nn.Module):
    """Module to calculate the (sparse) transformation matrix from the standard basis to the local
    basis."""

    def __init__(self) -> None:
        """Initializes the module."""
        super().__init__()
        self.local_basis_module = LocalBasisModule()

    def forward(
        self,
        n_basis: int,
        irreps_per_atom: np.ndarray[Irreps],
        pos: Tensor,
        atom_coo_indices: Tensor,
        atomic_numbers: Tensor | None = None,
        batch: Tensor | None = None,
        return_lframes: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Calculates the transformation matrix from the standard basis to the local basis.

        Args:
            n_basis: Total number of basis functions in the molecule.
            irreps_per_atom: Irreps of the basis functions per atom. Shape (n_atom,).
            pos: Positions of the atoms. Shape (n_atom, 3).
            atom_coo_indices: Indices that can be used to construct a per-atom block-diagonal sparse COOrdinate matrix,
                as returned by :func:`~mldft.ml.data.components.convert_transforms.add_atom_coo_indices`. Shape (2, n_basis).
            atomic_numbers (Tensor): The atomic numbers of the atoms in the molecule
            batch: Batch vector. Shape (n_atom,). Defaults to None.
            return_lframes: If True, the local frames are returned as well. Defaults to False.
        """
        bases = self.local_basis_module.forward(pos, atomic_numbers, batch)
        blocks = []
        # it might be possible to make this more efficient by computing even smaller blocks:
        # one for each tensor field, instead of one for each atom
        for irreps, basis in zip(irreps_per_atom, bases):
            blocks.append(pyscf_to_e3nn_local_frames_matrix(basis, irreps).to(pos.device))
        # construct sparse matrix
        values = torch.cat([block.flatten() for block in blocks])

        # mask out zero values, not needed in the sparse matrix.
        # there are plenty because we use bigger blocks than necessary (see comment above).
        non_zero_mask = values != 0
        mat = torch.sparse_coo_tensor(
            indices=atom_coo_indices[:, non_zero_mask],
            values=values[non_zero_mask],
            size=(n_basis,) * 2,
            is_coalesced=True,
        )

        if return_lframes:
            return mat, bases
        else:
            return mat

    def sample_forward(
        self, sample: OFData, return_lframes: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Wrapper around forward that takes an OFData object instead of the individual arguments.

        Args:
            sample: The sample.
            return_lframes: If True, the local frames are returned as well. Defaults to False.
        """
        return self.forward(
            sample.n_basis,
            sample.irreps_per_atom,
            sample.pos,
            sample.atom_coo_indices,
            sample.atomic_numbers,
            sample.batch,
            return_lframes=return_lframes,
        )


class LocalFramesTransformMatrixDense(nn.Module):
    """Module to calculate the (dense) transformation matrix from the standard basis to the local
    basis."""

    def __init__(self) -> None:
        """Initializes the module."""
        super().__init__()
        self.local_basis_module = LocalBasisModule()

    def forward(
        self,
        irreps_per_atom: np.ndarray[Irreps],
        pos: Tensor,
        atomic_numbers: Tensor | None = None,
        batch: Tensor | None = None,
        return_lframes: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Calculates the transformation matrix from the standard basis to the local basis.

        Args:
            irreps_per_atom: Irreps of the basis functions per atom. Shape (n_atom,).
            pos: Positions of the atoms. Shape (n_atom, 3).
            atomic_numbers (Tensor): The atomic numbers of the atoms in the molecule
            batch: Batch vector. Shape (n_atom,). Defaults to None.
            return_lframes: If True, the local frames are returned as well. Defaults to False.
        """
        bases = self.local_basis_module.forward(pos, atomic_numbers, batch)
        blocks = []
        # as in the sparse case, it might be possible to make this more efficient by computing even smaller blocks:
        # one for each tensor field, instead of one for each atom.
        for irreps, basis in zip(irreps_per_atom, bases):
            blocks.append(pyscf_to_e3nn_local_frames_matrix(basis, irreps).to(pos.device))

        if return_lframes:
            return torch.block_diag(*blocks), bases
        else:
            return torch.block_diag(*blocks)

    def sample_forward(
        self, sample: OFData, return_lframes: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Wrapper around forward that takes an OFData object instead of the individual arguments.

        Args:
            sample: The sample.
        """
        return self.forward(
            sample.irreps_per_atom,
            sample.pos,
            sample.atomic_numbers,
            sample.batch,
            return_lframes=return_lframes,
        )
