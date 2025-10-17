"""Submodule containing the neural network models.

Contains the main LightningModule in `mldft.ml.models.mldft_module` and the components
for the neural networks in `mldft.ml.models.components`.
"""


import warnings

for submodule in ["loss_function", "net"]:
    warnings.filterwarnings(
        "ignore",
        message=f".*Attribute '{submodule}' is an instance of `nn.Module` and is already saved during checkpointing.*",
    )
