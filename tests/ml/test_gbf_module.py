import pytest
import torch
from torch import Tensor

from mldft.ml.data.components.of_data import OFData
from mldft.ml.models.components.gbf_module import GBFModule


@pytest.mark.parametrize(
    "positions, edge_index, expected",
    [
        (
            Tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 0.0, 2.0]]),
            Tensor([[0, 1, 2, 3], [1, 2, 3, 0]]).int(),
            Tensor(
                [
                    [
                        0.24197073,
                        0.31944802,
                        0.37738323,
                        0.39894229,
                        0.37738323,
                        0.31944802,
                        0.24197073,
                        0.16401009,
                        0.09947712,
                        0.05399097,
                    ],
                    [
                        0.24197073,
                        0.31944802,
                        0.37738323,
                        0.39894229,
                        0.37738323,
                        0.31944802,
                        0.24197073,
                        0.16401009,
                        0.09947712,
                        0.05399097,
                    ],
                    [
                        0.00443185,
                        0.01139598,
                        0.02622189,
                        0.05399097,
                        0.09947714,
                        0.16401006,
                        0.24197073,
                        0.31944799,
                        0.37738323,
                        0.39894229,
                    ],
                    [
                        0.03274718,
                        0.06527553,
                        0.11643190,
                        0.18583974,
                        0.26543015,
                        0.33923998,
                        0.38797957,
                        0.39705965,
                        0.36361989,
                        0.29797831,
                    ],
                ]
            ),
        ),
    ],
)
def test_gbf_module_test_cases(positions: Tensor, edge_index: Tensor, expected: Tensor) -> None:
    """Tests the forward pass of the GBFModule with some test cases.

    Args:
        positions (Tensor): positions of the atoms
        edge_index (Tensor): edge index of the graph
        expected (Tensor): expected output of the forward pass
    """
    module = GBFModule(normalized=True)
    assert torch.allclose(module(OFData(pos=positions, edge_index=edge_index)), expected)


@pytest.mark.parametrize(
    "num_gaussians, normalized",
    [
        (5, True),
        (10, False),
        (20, True),
    ],
)
def test_gbf_module_shapes(num_gaussians: int, normalized: bool) -> None:
    """Test the shape of the output of the GBFModule.

    Args:
        num_gaussians (int): Number of gaussians to be used
        normalized (bool): Normalization flag of the gaussians
    """
    module = GBFModule(num_gaussians=num_gaussians, normalized=normalized)

    for i in range(10):
        num_atoms = int(torch.randint(1, 100, (1,)).item())
        num_edges = int(torch.randint(1, 100, (1,)).item())
        pos = torch.randn(num_atoms, 3)
        edge_index = torch.randint(0, num_atoms, (2, num_edges))
        edge_attr = module(OFData(pos=pos, edge_index=edge_index))
        assert edge_attr.shape == (num_edges, num_gaussians)


@pytest.mark.parametrize(
    "num_gaussians, integrated_area, scale",
    [
        (5, Tensor([0.5, 0.773373, 0.933193, 0.987776, 0.998650]), torch.ones(5)),
        (5, Tensor([0.5, 0.933193, 0.99865, 0.999997, 1]), torch.ones(5) * 0.5),
    ],
)
def test_gbf_normalisation(num_gaussians: int, integrated_area: Tensor, scale: Tensor) -> None:
    """Test the normalisation of the gaussians.

    Args:
        num_gaussians (int): Number of gaussians to be used
        integrated_area (Tensor): Integrated area of the gaussians in the boundaries [0, infintiy]
        scale (Tensor): Scale parameter of the gaussians
    """

    module = GBFModule(num_gaussians=num_gaussians, normalized=True)

    module.scale = torch.nn.Parameter(scale)

    num_atoms = 1000000

    # create a linear space of positions to get a linear distribution of distances to the first atom
    # Here 10 is a placeholder for the infinite integration boundary
    positions_x = torch.linspace(0, 10, num_atoms)
    positions_y = torch.zeros(num_atoms)
    positions_z = torch.zeros(num_atoms)

    # stack the positions into one tensor
    positions = torch.stack([positions_x, positions_y, positions_z], dim=1)

    # calculate the edge index where every node is only connected to the first atom
    edge_index_1 = torch.zeros(num_atoms)
    edge_index_2 = torch.linspace(0, num_atoms - 1, num_atoms)
    edge_index = torch.stack([edge_index_1, edge_index_2], dim=0).long()

    # calculate the output of the module
    output = module.forward(OFData(pos=positions, edge_index=edge_index))

    # test if the integrated area of the gaussians is  equal to the given integrated_area
    for i in range(num_gaussians):
        assert torch.allclose(10 / num_atoms * torch.sum(output[:, i]), integrated_area[i])
