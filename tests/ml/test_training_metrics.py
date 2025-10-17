import pytest
import torch
from torch_geometric.utils import unbatch

from mldft.ml.data.components.loader import OFLoader
from mldft.ml.models.components.loss_function import project_gradient_difference
from mldft.ml.models.components.toy_net import ToyNet
from mldft.ml.models.components.training_metrics import MAEGradient
from mldft.ml.models.mldft_module import MLDFTLitModule
from mldft.utils.utils import set_default_torch_dtype


@set_default_torch_dtype(torch.float64)
@pytest.mark.parametrize("batch_sizes", [[2, 5, 6, 32]])
@pytest.mark.parametrize("k_neighbors", [3, 5])
@pytest.mark.parametrize(
    "metric", [MAEGradient(mode="per molecule"), MAEGradient(mode="per electron")]
)
def test_metrics(metric, dummy_basis_info, dummy_dataset_torch, batch_sizes, k_neighbors):
    """Tests if the shape of the toy net output is correct."""
    loss = torch.zeros(len(batch_sizes), dtype=torch.float64)
    net = ToyNet(dummy_basis_info, k_neighbors)
    module = MLDFTLitModule(
        net,
        basis_info=dummy_basis_info,
        optimizer=None,
        scheduler=None,
        loss_function=None,
        target_key="kin",
        compile=False,
    )
    # Test that the metric is independent of the batchsize
    for i, batch_size in enumerate(batch_sizes):
        dataloader = OFLoader(
            dummy_dataset_torch, batch_size=batch_size, shuffle=False, num_workers=0
        )
        metric.reset()
        for batch in dataloader:
            pred_energy, pred_gradients, pred_diff = module.forward(batch)
            projected_gradient_diff = project_gradient_difference(pred_gradients, batch)
            metric.update(batch, pred_energy, projected_gradient_diff, pred_diff)
        loss[i] = metric.compute()
    assert pytest.approx(loss) == loss[0] * torch.ones(len(batch_sizes))
    # check if the shapes are right
    assert pred_energy.shape == batch.energy_label.shape
    assert pred_gradients.shape == batch.gradient_label.shape


@set_default_torch_dtype(torch.float64)
@pytest.mark.parametrize("batch_sizes", [[2, 5, 6, 32]])
@pytest.mark.parametrize("k_neighbors", [3, 5])
def test_metrics2(dummy_basis_info, dummy_dataset_torch, batch_sizes, k_neighbors):
    """Tests if the shape of the projection is correct.

    And if the projection is calculated in the right way
    """
    loss = torch.zeros(len(batch_sizes), dtype=torch.float64)
    net = ToyNet(dummy_basis_info, k_neighbors)

    # Test that the metric is independent of the batchsize
    for i, batch_size in enumerate(batch_sizes):
        dataloader = OFLoader(
            dummy_dataset_torch, batch_size=batch_size, shuffle=False, num_workers=0
        )
        for batch in dataloader:
            out = net.forward(batch)
            pred_energy, pred_gradients = out[0], out[1]
            data_list = batch.to_data_list()
            prediction_gradient_list = unbatch(pred_gradients, batch.coeffs_batch)
            for data, prediction_gradient1 in zip(data_list, prediction_gradient_list):
                dual_basis_integrals = data.dual_basis_integrals
                delta = data.gradient_label - prediction_gradient1
                proj = torch.einsum(
                    "i,j,j -> i", dual_basis_integrals, dual_basis_integrals, delta
                ) / torch.dot(dual_basis_integrals, dual_basis_integrals)
                proj_2 = (
                    dual_basis_integrals
                    * torch.dot(dual_basis_integrals, delta)
                    / torch.dot(dual_basis_integrals, dual_basis_integrals)
                )
                assert proj.shape == delta.shape
                assert proj.detach() == pytest.approx(proj_2.detach())
