"""Implements the full model as described in [M-OFDFT]_."""

from typing import Tuple, Union

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from pyscf.data.elements import ELEMENTS
from torch import Tensor
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.nn.pool import global_add_pool

from mldft.ml.data.components.of_batch import OFBatch
from mldft.ml.data.components.of_data import OFData
from mldft.ml.models.components.atom_ref import AtomRef
from mldft.ml.models.components.dimension_wise_rescaling import DimensionWiseRescaling
from mldft.ml.models.components.g3d_stack import G3DStack
from mldft.ml.models.components.gbf_module import GaussianLayer, GBFModule
from mldft.ml.models.components.initial_guess_delta_module import (
    InitialGuessDeltaModule,
)
from mldft.ml.models.components.mlp import MLP
from mldft.ml.models.components.node_embedding import NodeEmbedding
from mldft.utils.log_utils.logging_mixin import LoggingMixin

# set tensorframes to None if not available
try:
    from tensorframes.lframes import LFrames
except ImportError:
    LFrames = None


class MLPStack(nn.Module):
    def __init__(self, mlp_class: nn.Sequential, n_mlps: int, **mlp_kwargs) -> None:
        super().__init__()
        self.n_mlps = n_mlps
        # List of learnable scalars
        self.weights = nn.Parameter(torch.ones(n_mlps))
        self.mlps = nn.ModuleList([mlp_class(**mlp_kwargs) for _ in range(n_mlps)])

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        out = self.weights[0] * self.mlps[0](x[0], batch)
        for i in range(1, self.n_mlps):
            out = out + self.weights[i] * self.mlps[i](x[i], batch)

        return out / self.n_mlps


class Graphformer(nn.Module, LoggingMixin):
    """The Graphformer module as described in [M-OFDFT]_."""

    def __init__(
        self,
        edge_mlp: MLP,
        energy_mlp: MLP | MLPStack,
        gbf_module: GBFModule,
        node_embedding_module: NodeEmbedding,
        gnn_module: G3DStack,
        atom_ref_module: AtomRef,
        initial_guess_module: InitialGuessDeltaModule,
        dimension_wise_rescaling_module: DimensionWiseRescaling,
        final_energy_factor: float = 1.0,
        final_norm_layer: nn.Module = None,
    ) -> None:
        """Initializes the Graphformer class.

        Args:
            edge_mlp (MLP): The MLP predicting edge attributes as input for the G3D Layer.
            energy_mlp (MLP): The MLP predicting the energy per atom.
            gbf_module (GBFModule): The GBF module.
            node_embedding_module (NodeEmbedding): The node embedding module.
            gnn_module (G3DStack): The stack of G3DLayers, which make up the main part of the module. Can be replaced
            by any Graph NN module with the same signature.
            atom_ref_module (AtomRef): The atomic reference module.
            initial_guess_module (InitialGuessDeltaModule): The Module mapping the final node_features to the initial
            guess differences.
            dimension_wise_rescaling_module: The dimension wise rescaling module.
        """
        super().__init__()
        self.edge_mlp = edge_mlp
        self.gbf_module = gbf_module
        self.node_embedding_module = node_embedding_module
        self.gnn_module = gnn_module
        self.energy_mlp = energy_mlp
        if isinstance(energy_mlp, MLPStack):
            n_readouts = gnn_module.n_layers / gnn_module.energy_readout_every
            assert (
                n_readouts == energy_mlp.n_mlps
            ), f"Energy MLP stack must have the same number of MLPs ({energy_mlp.n_mlps}) as the number of configured readouts ({n_readouts})."
        elif isinstance(energy_mlp, MLP):
            assert (
                gnn_module.energy_readout_every == gnn_module.n_layers
            ), "Energy should only be readout at the last layer when using a single MLP as the energy readout."
        self.atom_ref_module = atom_ref_module
        self.initial_guess_module = initial_guess_module
        self.dimension_wise_rescaling_module = dimension_wise_rescaling_module
        self.final_energy_factor = final_energy_factor
        self.final_norm_layer = final_norm_layer

    def forward(self, batch: Union[OFData, OFBatch]) -> Tuple[Tensor, Tensor]:
        """Calculates the forward pass of the module according to the Figure 6 in [M-OFDFT]_.

        Args:
            batch (OFData): The batch / data object containing the input data.

        Returns:
            Tuple[Tensor, Tensor]: The energy and the initial guess delta, in this order.
        """
        edge_index = batch.edge_index

        # upper part of figure 6 in the M-OFDFT paper
        gbf_embedding, length = self.gbf_module(batch)
        # these are scalar attributes
        g3d_edge_attr = self.edge_mlp(gbf_embedding, batch.batch)

        # middle row of figure 6
        coeffs_rescaled = self.dimension_wise_rescaling_module(batch)
        node_features = self.node_embedding_module(
            coeffs=coeffs_rescaled,
            atom_ind=batch.atom_ind,
            basis_function_ind=batch.basis_function_ind,
            coeff_ind_to_node_ind=batch.coeff_ind_to_node_ind,
            n_basis_dim_per_atom=batch.n_basis_per_atom,
            distance_embedding=gbf_embedding,
            edge_index=edge_index,
            length=length,
        )
        if getattr(batch, "lframes", None) is not None and LFrames is not None:
            lframes = LFrames(matrices=batch.lframes.reshape(-1, 3, 3))
        else:
            # If the gnn_module requires lframes it will raise an error
            lframes = None
        node_features = self.gnn_module(
            x=node_features,
            edge_index=edge_index,
            batch=batch.batch,
            edge_attr=g3d_edge_attr,
            length=length,
            lframes=lframes,
        )
        if hasattr(self, "final_norm_layer") and self.final_norm_layer is not None:
            if isinstance(self.final_norm_layer, LayerNorm):
                node_features = self.final_norm_layer(node_features, batch.batch)
            else:
                node_features = self.final_norm_layer(node_features)
        energies_per_atom = self.energy_mlp(node_features, batch.batch)
        # If we use an MLP the output will still have the first n_readouts dimension with size 1
        if energies_per_atom.ndim == 3:
            energies_per_atom.squeeze_(0)
        energies = global_add_pool(energies_per_atom, batch.batch)[:, 0]

        if hasattr(self, "final_energy_factor"):  # for compatibility with older models
            energies *= self.final_energy_factor

        # lower part of figure 6
        energies_atom_ref = self.atom_ref_module.sample_forward(batch)
        self.log(energy_atom_ref=energies_atom_ref.sum(), energy_g3d=energies.sum())
        energies += energies_atom_ref
        # output branch initial guess
        initial_guess_delta = self.initial_guess_module(
            x=node_features[-1],
            basis_function_ind=batch.basis_function_ind,
            coeff_ind_to_node_ind=batch.coeff_ind_to_node_ind,
            batch=batch.batch,
        )

        return energies, initial_guess_delta

    def get_distance_embeddings(self, distances: Tensor) -> Tuple[Tensor, Tensor]:
        """Get the distance embeddings for the given distances.

        Args:
            distances (Tensor): The distances for which to calculate the embeddings.

        Returns:
            gbf_embedding, g3d_edge_attr: The distance embeddings, shapes (n_distances, n_embedding_dims) and
                (n_distances, 1), respectively.
        """

        assert distances.dim() == 1, f"Distances must be 1D but got shape {distances.shape}."

        # create positions: first atom at origin, other atom on x-axis at given distances
        pos = torch.stack(
            [distances, torch.zeros_like(distances), torch.zeros_like(distances)], dim=1
        )  # (n_distances, 3)
        pos = torch.cat([pos.new_zeros((1, 3)), pos], dim=0)  # (n_distances + 1, 3)

        # create edge index connecting the first atom with all others
        device = distances.device
        edge_index = torch.stack(
            [
                torch.zeros_like(distances, dtype=torch.long),
                torch.arange(1, distances.shape[0] + 1, device=device),
            ],
            dim=0,
        )
        dummy_sample = OFData(
            pos=pos,
            edge_index=edge_index,
            atom_ind=torch.zeros_like(
                pos[:, 0], dtype=torch.long
            ),  # dummy atom indices: all H (probably, depends on basis_info)
        )

        gbf_embedding, _ = self.gbf_module(dummy_sample)
        g3d_edge_attr = self.edge_mlp(gbf_embedding)

        return gbf_embedding, g3d_edge_attr

    def plot_distance_embeddings(
        self, max_distance: float = 10.0, n_distances: int = 1000
    ) -> plt.Figure:
        """Plot the distance embeddings for a range of distances.

        Args:
            max_distance (float): The maximum distance to consider.
            n_distances (int): The number of distances to consider.

        Returns:
            plt.Figure: The plot.
        """
        device = next(self.parameters()).device  # infer device from first parameter
        distances = torch.linspace(0, max_distance, n_distances, device=device)
        gbf_embedding, g3d_edge_attr = self.get_distance_embeddings(distances)

        distances = distances.cpu().numpy()
        gbf_embedding = gbf_embedding.detach().cpu().numpy()
        g3d_edge_attr = g3d_edge_attr.detach().cpu().numpy()

        cmap = plt.cm.viridis

        n_ax = 2
        with_edge_type = False
        if isinstance(self.gbf_module, GaussianLayer):
            n_ax += 1
            with_edge_type = True

        fig, axs = plt.subplots(n_ax, 1, figsize=(15, 5 * n_ax))
        # plot gbf_embedding as multiple lines in upper plot, colored by their index using the viridis colormap
        for i in range(gbf_embedding.shape[1]):
            axs[0].plot(
                distances,
                gbf_embedding[:, i],
                color=cmap(i / gbf_embedding.shape[1]),
                linewidth=1,
            )
        axs[0].set_ylabel("Gaussian embedding value")
        axs[0].set_xlabel("Distance [Bohr]")
        axs[0].set_xlim(0, max_distance)

        axs[1].plot(distances, g3d_edge_attr)
        axs[1].set_ylabel("Edge attribute value")
        axs[1].set_xlabel("Distance [Bohr]")
        axs[1].set_xlim(0, max_distance)

        if with_edge_type:
            ax = axs[2]
            basis_info = self.gbf_module.basis_info

            edge_type_labels = [
                f"{ELEMENTS[basis_info.atomic_numbers[i]]}-{ELEMENTS[basis_info.atomic_numbers[j]]}"
                for i in range(basis_info.n_types)
                for j in range(basis_info.n_types)
            ]
            edge_type_labels = edge_type_labels[
                : self.gbf_module.n_edge_types
            ]  # will be shorter in undirected case
            x = torch.arange(len(edge_type_labels))
            ax.scatter(
                x,
                self.gbf_module.mul.weight.view(-1).detach().cpu().numpy(),
                label="mul",
                color="red",
            )
            ax.scatter(
                x,
                self.gbf_module.bias.weight.view(-1).detach().cpu().numpy(),
                label="bias",
                color="blue",
            )
            ax.set_xticks(range(len(edge_type_labels)))
            ax.set_xticklabels(edge_type_labels, rotation=45)
            ax.legend()
            ax.set_xlabel("Edge type")
            ax.set_axisbelow(True)
            ax.grid(True)

        plt.tight_layout()
        return fig
