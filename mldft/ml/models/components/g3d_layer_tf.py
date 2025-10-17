"""Graphormer layer with scalar distance features as attention bias.

Implements the G3D layer as described in [M-OFDFT]_, based on [Graphormer]_.
"""

import math
import warnings
from typing import Optional

import torch
from tensorframes.lframes import LFrames
from tensorframes.nn.tfmessage_passing import TFMessagePassing
from tensorframes.reps import Irreps
from torch import Tensor, nn
from torch.nn import Linear
from torch_geometric.nn.norm.layer_norm import LayerNorm
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax

from mldft.ml.models.components.mlp import MLP
from mldft.ml.models.components.node_embedding import smooth_falloff
from mldft.utils.log_utils.logging_mixin import LoggingMixin


class G3DLayerTF(TFMessagePassing, LoggingMixin):
    """The G3D layer as described in [M-OFDFT]_ adapted using the tensorframes formalism. Keys and
    values are transformed into the local frames of the receiving node. The TFMessagePassing class
    handles all trafos automatically.

    ``out_channels`` are computed by dividing the input dimension by the number of heads.

    Based on :class:`~torch_geometric.nn.conv.transformer_conv.TransformerConv` as implemented in :mod:`torch_geometric.nn.conv.transformer_conv`.
    """

    # Needed for torch_geometric.compile
    propagate_type = {
        "query": Tensor,
        "key": Tensor,
        "value": Tensor,
        "edge_attr": Tensor,
        "lframes": LFrames,
        "length": Tensor,
    }

    def __init__(
        self,
        in_reps: Irreps,
        heads: int = 32,
        edge_dim: int = 1,
        dropout: float = 0.0,
        attention_weight_dropout: float = 0.0,
        mlp_hidden_dim: Optional[int] = None,
        mlp_activation: torch.nn.Module = torch.nn.GELU,
        mlp_norm_layer: torch.nn.Module = None,
        norm_layer_class: torch.nn.Module = LayerNorm,
        activation_dropout: float = 0.0,
        cutoff: float = None,
        cutoff_start: float = 0.0,
        **kwargs,
    ):
        """Initialize the G3DLayer.

        Args:
            in_reps Irreps: Irreps that should be used during message passing.
            heads (int, optional): Number of multi-head-attentions.
                (default: `1`)
            edge_dim (int): Edge feature dimensionality (in case
                there are any). Edge features are added to the attention weights before
                applying the soft(arg)max. (default: `1`)
            dropout (float, optional): Dropout probability of the MLP. Defaults to 0.0.
            attention_weight_dropout (float, optional): Dropout probability of the attention weights. Defaults to 0.0.
            mlp_hidden_dim (int, optional): Hidden dimensionality of the MLP. If None, defaults to `in_channels`.
            mlp_activation (torch.nn.Module, optional): Activation function of the MLP. Defaults to `torch.nn.GELU()`.
            activation_dropout (float, optional): Dropout probability of the activation function. Defaults to 0.0.
            **kwargs (optional): Additional arguments of `torch_geometric.nn.conv.MessagePassing`.

        Raises:
            ValueError: If the number of heads does not divide the number of input channels.
        """
        kwargs.setdefault("aggr", "add")
        super().__init__(
            node_dim=0,
            params_dict={
                "key": {"type": "local", "rep": in_reps},
                "value": {"type": "local", "rep": in_reps},
            },
            **kwargs,
        )
        self.cutoff = cutoff
        self.cutoff_start = cutoff_start
        self.in_reps = in_reps
        in_channels = in_reps.dim
        self.in_channels = in_channels
        self.heads = heads

        if in_channels % heads != 0:
            raise ValueError("Number of heads must divide in_channels.")

        self.channels_per_head = in_channels // heads
        self.edge_dim = edge_dim
        self.mlp_hidden_dim = mlp_hidden_dim if mlp_hidden_dim is not None else in_channels

        # heads * channel_per_head = in_channels
        # Use one large linear layer to compute query, key and value more efficiently
        self.linear_in = Linear(in_channels, 3 * in_channels)
        self.linear_out = Linear(in_channels, in_channels)

        self.norm_1 = norm_layer_class(self.in_channels)
        self.norm_2 = norm_layer_class(self.in_channels)

        self.dropout = torch.nn.Dropout(dropout)
        self.attention_weight_dropout = torch.nn.Dropout(attention_weight_dropout)
        # Two layer mlp, to hidden_dim and back to in_channels
        self.mlp = MLP(
            in_channels=self.in_channels,
            hidden_channels=[self.mlp_hidden_dim, self.in_channels],
            activation_layer=mlp_activation,
            dropout=activation_dropout,
            disable_dropout_last_layer=True,
            disable_activation_last_layer=True,
            norm_layer=mlp_norm_layer,
        )
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the linear layers (query, key, value, and output linear layers)
        nn.init.xavier_uniform_(self.linear_in.weight)
        nn.init.constant_(self.linear_in.bias, 0)
        nn.init.xavier_uniform_(self.linear_out.weight)
        nn.init.constant_(self.linear_out.bias, 0)
        # Initialize the MLP, if it has learnable parameters
        if hasattr(self.mlp, "reset_parameters"):
            self.mlp.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        batch: Tensor,
        lframes: LFrames,
        edge_attr: OptTensor = None,
        length: OptTensor = None,
    ) -> Tensor:
        """Runs the forward pass of the module.

        The forward pass is defined as:
        x = MHAtt(LN(input) + edge_attr) + input
        output = x + MLP(LN(x))

        Args:
            x (torch.Tensor): The input node
                features.
            edge_index (torch.Tensor): The edge indices.
            batch (torch.Tensor): Batch assigning each node to a specific graph.
            lframes (LFrames): The LFrames object containing the local frames.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: `None`)
            length (torch.Tensor, optional): The length of the edges sorted as in the edge index.

        Returns:
            torch.Tensor: The output node features.
        """
        assert lframes is not None, "lframes must not be None"

        if edge_attr is None:
            warnings.warn("Edge_attr is None. This is not recommended.")
            edge_attr = torch.zeros((edge_index.shape[1], self.edge_dim), device=x.device)

        # Attention Block
        if hasattr(self, "layer_norm_1"):
            out_normalized = self.layer_norm_1(x, batch)
        elif isinstance(self.norm_1, LayerNorm):
            out_normalized = self.norm_1(x, batch)
        else:
            out_normalized = self.norm_1(x)
        # Chunk the linear layer output into query, key and value
        query, key, value = self.linear_in(out_normalized).chunk(3, dim=-1)
        attention_output = self.propagate(
            edge_index,
            query=query,
            key=key,
            value=value,
            edge_attr=edge_attr,
            size=None,
            lframes=lframes,
            length=length,
        )
        attention_output = attention_output.view(-1, self.heads * self.channels_per_head)
        attention_output = self.linear_out(attention_output)
        attention_output = self.dropout(attention_output)
        out_skip_links = attention_output + x

        # MLP Block
        if hasattr(self, "layer_norm_2"):
            out_normalized = self.layer_norm_2(out_skip_links, batch)
        elif isinstance(self.norm_2, LayerNorm):
            out_normalized = self.norm_2(out_skip_links, batch)
        else:
            out_normalized = self.norm_2(out_skip_links)

        out_mlp = self.mlp(out_normalized, batch)
        out_mlp = self.dropout(out_mlp)
        self.log(out_mlp=out_mlp, out_skip_links=out_skip_links)
        out = out_mlp + out_skip_links
        return out

    def message(
        self,
        query_i: Tensor,
        key_j: Tensor,
        value_j: Tensor,
        edge_attr: Tensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
        length: OptTensor = None,
    ) -> Tensor:
        """Message function of the G3D layer. Computes the attention weights of each edge, added
        with the according edge_attr.

        Args:
            query_i: query edge tensor of shape (E, heads, channels_per_head)
            key_j: key edge tensor of shape (E, heads, channels_per_head)
            value_j: value edge tensor of shape (E, heads, channels_per_head)
            edge_attr: edge features
            index: the indices describing where edges end
            ptr: pointer to indicate where graph in a batch ends and starts
            size_i:The dimension in which the softmax normalizes.
            length: The length of the edges sorted as in the edge index.

        Returns:
            Tensor: The output tensor.
        """
        query_i = query_i.contiguous().view(-1, self.heads, self.channels_per_head)
        key_j = key_j.contiguous().view(-1, self.heads, self.channels_per_head)
        value_j = value_j.contiguous().view(-1, self.heads, self.channels_per_head)

        alpha = self.compute_attention(query_i, key_j, edge_attr, index, ptr, size_i)
        if hasattr(self, "cutoff") and self.cutoff is not None:
            alpha = alpha * smooth_falloff(length, self.cutoff, self.cutoff_start)
        alpha = self.attention_weight_dropout(alpha)
        out = value_j * alpha.view(-1, self.heads, 1)
        return out

    def compute_attention(
        self,
        query_i: Tensor,
        key_j: Tensor,
        edge_attr: Tensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ):
        """Compute the attention weights."""
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.channels_per_head) + edge_attr
        alpha = softmax(alpha, index, ptr, size_i)
        return alpha

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

    def __repr__(self) -> str:
        """Representation of the G3D layer."""
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"heads={self.heads}), edge_feature_dim={self.edge_dim})"
        )


class G3DLayerMul(G3DLayerTF):
    """G3D layer with multiplicative attention bias."""

    def compute_attention(
        self,
        query_i: Tensor,
        key_j: Tensor,
        edge_attr: Tensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ):
        """Compute the attention weights."""
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.channels_per_head) * edge_attr
        alpha = softmax(alpha, index, ptr, size_i)
        return alpha


class G3DLayerSilu(G3DLayerTF):
    """G3D layer with SiLU activation function."""

    def compute_attention(
        self,
        query_i: Tensor,
        key_j: Tensor,
        edge_attr: Tensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ):
        """Compute the attention weights."""
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.channels_per_head) + edge_attr
        alpha = torch.nn.functional.silu(alpha)
        return alpha


class G3DLayerMulSilu(G3DLayerTF):
    """G3D layer with SiLU activation function."""

    def compute_attention(
        self,
        query_i: Tensor,
        key_j: Tensor,
        edge_attr: Tensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ):
        """Compute the attention weights."""
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.channels_per_head) * edge_attr
        alpha = torch.nn.functional.silu(alpha)
        return alpha
