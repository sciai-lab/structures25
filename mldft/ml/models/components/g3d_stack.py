"""Module to wrap a stack of :class:`~mldft.ml.models.components.g3d_layer.G3DLayer`."""
import torch
import torch.nn as nn
from torch import Tensor

from mldft.ml.models.components.g3d_layer import G3DLayer

# set tensorframes to None if not available
try:
    from tensorframes.lframes import LFrames

    from mldft.ml.models.components.g3d_layer_tf import G3DLayerTF
except ImportError:
    G3DLayerTF = None
    LFrames = None


class G3DStack(nn.Module):
    """Module to wrap a stack of G3D layers."""

    def __init__(
        self,
        g3d_class: G3DLayer,
        n_layers: int,
        energy_readout_every: int | None = None,
        **g3d_kwargs,
    ) -> None:
        """The G3DStack module is a stack of
        :class:`~mldft.ml.models.components.g3d_layer.G3DLayer`.

        Args:
            n_layers (int): number of G3D layers.
            energy_readout_every (int, optional): The frequency of energy readout. If `None`, the energy is read out
                every `n_layers` which means only the last layer will be read out.
            **g3d_kwargs: Arguments of `G3DLayer`.
        """
        super().__init__()
        self.n_layers = n_layers
        self.energy_readout_every = (
            energy_readout_every if energy_readout_every is not None else n_layers
        )
        self.g3d_layers = nn.ModuleList([g3d_class(**g3d_kwargs) for _ in range(n_layers)])

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        lframes: LFrames = None,
        edge_attr: Tensor = None,
        length: Tensor = None,
    ) -> Tensor:
        """Calculates the forward pass of the module by going through all the G3D layers.

        Args:
            x (Tensor): The input node
                features.
            edge_index (Tensor): The edge indices.
            batch (Tensor): Batch assigning each node to a specific graph.
            lframes (LFrames, optional): The LFrames object containing the local frames. (default: `None`)
            edge_attr (Tensor, optional): The edge features.
                (default: `None`)
            length (Tensor, optional): The length of the edges sorted as in the edge index.

        Returns:
            Tensor: The output node features.
        """

        inter_nodes = []
        energy_readout_every = (
            self.energy_readout_every if hasattr(self, "energy_readout_every") else self.n_layers
        )
        for i in range(self.n_layers):
            g3d_layer = self.g3d_layers[i]
            if isinstance(g3d_layer, G3DLayer):
                x = self.g3d_layers[i](
                    x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr, length=length
                )
            elif isinstance(g3d_layer, G3DLayerTF):
                x = self.g3d_layers[i](
                    x=x,
                    edge_index=edge_index,
                    batch=batch,
                    lframes=lframes,
                    edge_attr=edge_attr,
                    length=length,
                )
            else:
                raise NotImplementedError("G3DLayer class not recognized.")

            if (i + 1) % energy_readout_every == 0:
                inter_nodes.append(x)

        return torch.stack(inter_nodes)
