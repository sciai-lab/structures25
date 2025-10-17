import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from mldft.ml.models.components.loss_function import (
    CoefficientLoss,
    EnergyGradientLoss,
    EnergyLoss,
    WeightedLoss,
    project_gradient_difference,
)


@pytest.mark.parametrize("batch_size", [4, 16])
def test_loss_functions_shape(batch_size):
    """Tests the shape of the output of the loss function and checks if the loss function is zero
    for the same input and target."""

    # Define a dataset to be used in the dataloader

    dataset = []

    for i in range(100):
        # define random length coefficients
        length = int(torch.randint(1, 20, (1,)))

        kin_energy = torch.rand(1)
        has_energy_label = torch.randint(1, (1,), dtype=torch.bool)
        kin_gradients = torch.randn((length,))
        basis_integrals = torch.randn((length,))

        ground_state = torch.randn((length,))
        coeffs = ground_state + 1

        data = Data(
            energy_label=kin_energy,
            has_energy_label=has_energy_label,
            gradient_label=kin_gradients,
            basis_integrals=basis_integrals,
            dual_basis_integrals=basis_integrals,
            ground_state_coeffs=ground_state,
            coeffs=coeffs,
        )

        dataset.append(data)

    # create a dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, follow_batch=["coeffs"])

    # create Loss function
    module = WeightedLoss(
        energy_loss=dict(weight=0.3, loss=EnergyLoss(loss_function=nn.L1Loss(reduction="none"))),
        gradient_loss=dict(
            weight=0.3, loss=EnergyGradientLoss(loss_function=nn.L1Loss(reduction="none"))
        ),
        coefficient_loss=dict(
            weight=0.3, loss=CoefficientLoss(loss_function=nn.L1Loss(reduction="none"))
        ),
    )

    # check for 10 batches
    for batch in dataloader:
        # calculate the loss
        projected_gradient_difference = project_gradient_difference(batch.gradient_label, batch)
        weight_dict, loss_dict = module.forward(
            batch,
            pred_energy=batch.energy_label,
            projected_gradient_difference=projected_gradient_difference,
            pred_diff=batch.coeffs - batch.ground_state_coeffs,
        )
        assert weight_dict.keys() == loss_dict.keys()
        for loss in loss_dict.values():
            assert loss == 0.0
            assert loss.shape == torch.Size([])
        total_loss = sum([weight_dict[key] * loss_dict[key] for key in loss_dict.keys()])
        # check if the loss is zero and if the shape is correct
        assert total_loss == 0.0
        assert total_loss.shape == torch.Size([])


if __name__ == "__main__":
    test_loss_functions_shape()
