from typing import Callable, List, Optional

import torch
from torch_geometric.nn.norm import LayerNorm


class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Note:
        Adapted from :class:`torchvision.ops.MLP`, the only difference being the option to disable dropout in the
        last layer, and the fact that no dropout layers are added if the dropout probability is 0.0.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        inplace (bool, optional): Parameter for the activation layer, which can optionally do the operation in-place.
            Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
        dropout_last_layer (bool): Whether to use dropout in the last layer. Default: ``True``
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.0,
        disable_norm_last_layer: bool = False,
        disable_activation_last_layer: bool = False,
        disable_dropout_last_layer: bool = False,
    ):
        """Initializes the MLP module."""
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}
        assert 0 <= dropout < 1, f"Dropout probability must be in [0,1), got {dropout=}."

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim
        # Configure last layer
        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        if not disable_norm_last_layer and norm_layer is not None:
            layers.append(norm_layer(hidden_channels[-1]))
        if not disable_activation_last_layer:
            layers.append(activation_layer(**params))
        if not disable_dropout_last_layer:
            layers.append(torch.nn.Dropout(dropout, **params))
        super().__init__(*layers)

    def forward(self, x: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """Calculates the forward pass of the module.

        Args:
            x (torch.Tensor): The input tensor.
            batch (torch.Tensor): Batch tensor.

        Returns:
            torch.Tensor: The output tensor.
        """

        for layer in self:
            if isinstance(layer, LayerNorm) and layer.mode == "graph":
                assert batch is not None, "Batch tensor must be provided for LayerNorm."
                x = layer(x, batch)
            else:
                x = layer(x)

        return x
