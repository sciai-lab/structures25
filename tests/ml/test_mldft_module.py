import pytest
import torch
import torch.nn as nn

from mldft.ml.data.components.loader import OFLoader
from mldft.ml.models.components.loss_function import (
    CoefficientLoss,
    EnergyGradientLoss,
    EnergyLoss,
    WeightedLoss,
)
from mldft.ml.models.components.toy_net import ToyNet
from mldft.ml.models.mldft_module import MLDFTLitModule
from mldft.utils.utils import set_default_torch_dtype


@set_default_torch_dtype(torch.float64)
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("k_neighbors", [3, 5])
def test_mldft_module(dummy_basis_info, dummy_dataset_torch, batch_size, k_neighbors):
    """Tests the model_step function of the MLDFTLitModule class."""
    dataloader = OFLoader(dummy_dataset_torch, batch_size=batch_size, shuffle=False, num_workers=0)

    batch = next(iter(dataloader))

    net = ToyNet(dummy_basis_info, k_neighbors)
    loss = WeightedLoss(
        energy_loss=dict(weight=0.3, loss=EnergyLoss(loss_function=nn.L1Loss(reduction="none"))),
        gradient_loss=dict(
            weight=0.3, loss=EnergyGradientLoss(loss_function=nn.L1Loss(reduction="none"))
        ),
        coefficient_loss=dict(
            weight=0.3, loss=CoefficientLoss(loss_function=nn.L1Loss(reduction="none"))
        ),
    )

    model = MLDFTLitModule(
        net=net,
        optimizer=None,
        scheduler=None,
        loss_function=loss,
        target_key="kin",
        compile=True,
        basis_info=dummy_basis_info,
    )

    test = model.forward(batch)

    assert test[0].shape == batch.energy_label.shape
    assert test[1].shape == batch.gradient_label.shape
    assert test[2].shape == batch.coeffs.shape
