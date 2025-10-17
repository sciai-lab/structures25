import pytest
import torch
from e3nn.o3 import Irreps
from torch import Tensor

from mldft.ml.data.components.basis_transforms import MasterTransformation
from mldft.ml.data.components.convert_transforms import AddAtomCooIndices, ToTorch
from mldft.ml.data.components.dataset import OFDataset
from mldft.ml.models.components.local_frames_module import (
    LocalBasisModule,
    LocalFramesModule,
    LocalFramesTransformMatrixDense,
    LocalFramesTransformMatrixSparse,
)
from mldft.utils.utils import set_default_torch_dtype


@pytest.mark.parametrize(
    "pos, batch, expected",
    [
        (
            Tensor(
                [
                    [-1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 2.0, 1.0],
                ]
            ),
            None,
            Tensor(
                [
                    [
                        [0.89442718, 0.44721359, 0.00000000],
                        [-0.35856858, 0.71713716, 0.59761435],
                        [0.26726127, -0.53452253, 0.80178374],
                    ],
                    [
                        [-0.57735026, 0.57735026, 0.57735026],
                        [-0.77151674, -0.61721337, -0.15430334],
                        [0.26726124, -0.53452247, 0.80178374],
                    ],
                    [
                        [0.57735026, -0.57735026, -0.57735026],
                        [-0.77151674, -0.61721337, -0.15430334],
                        [-0.26726124, 0.53452247, -0.80178374],
                    ],
                ]
            ),
        ),
        (
            Tensor(
                [
                    [1.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 2.0, 1.0],
                ]
            ),
            None,
            Tensor(
                [
                    [
                        [-0.57735026, 0.57735026, 0.57735026],
                        [-0.77151674, -0.61721337, -0.15430334],
                        [0.26726124, -0.53452247, 0.80178374],
                    ],
                    [
                        [0.89442718, 0.44721359, 0.00000000],
                        [-0.35856858, 0.71713716, 0.59761435],
                        [0.26726127, -0.53452253, 0.80178374],
                    ],
                    [
                        [0.57735026, -0.57735026, -0.57735026],
                        [-0.77151674, -0.61721337, -0.15430334],
                        [-0.26726124, 0.53452247, -0.80178374],
                    ],
                ]
            ),
        ),
        (
            Tensor(
                [
                    [-1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 2.0, 1.0],
                    [-1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 2.0, 1.0],
                ]
            ),
            Tensor([0, 0, 0, 1, 1, 1]),
            Tensor(
                [
                    [
                        [0.89442718, 0.44721359, 0.00000000],
                        [-0.35856858, 0.71713716, 0.59761435],
                        [0.26726127, -0.53452253, 0.80178374],
                    ],
                    [
                        [-0.57735026, 0.57735026, 0.57735026],
                        [-0.77151674, -0.61721337, -0.15430334],
                        [0.26726124, -0.53452247, 0.80178374],
                    ],
                    [
                        [0.57735026, -0.57735026, -0.57735026],
                        [-0.77151674, -0.61721337, -0.15430334],
                        [-0.26726124, 0.53452247, -0.80178374],
                    ],
                    [
                        [0.89442718, 0.44721359, 0.00000000],
                        [-0.35856858, 0.71713716, 0.59761435],
                        [0.26726127, -0.53452253, 0.80178374],
                    ],
                    [
                        [-0.57735026, 0.57735026, 0.57735026],
                        [-0.77151674, -0.61721337, -0.15430334],
                        [0.26726124, -0.53452247, 0.80178374],
                    ],
                    [
                        [0.57735026, -0.57735026, -0.57735026],
                        [-0.77151674, -0.61721337, -0.15430334],
                        [-0.26726124, 0.53452247, -0.80178374],
                    ],
                ]
            ),
        ),
    ],
)
def test_local_basis_module(pos, batch, expected):
    """Test the function local_basis_module."""
    b_module = LocalBasisModule(ignore_hydrogen=False)
    torch.testing.assert_close(b_module.forward(pos, batch=batch), expected[:, [2, 0, 1]])


@pytest.mark.parametrize(
    "coeffs, irreps, pos, atomic_numbers, batch, expected",
    [
        (
            [
                torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64),
                torch.tensor([1.0], dtype=torch.float64),
                torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64),
            ],
            [Irreps("0e+1e"), Irreps("0e"), Irreps("0e+1e")],
            torch.tensor(
                [
                    [-1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 2.0, 1.0],
                ],
                dtype=torch.float64,
            ),
            Tensor([2, 2, 2]),  # no hydrogen
            None,
            [
                Tensor([1.0, 2.13808882, 3.13049495, 3.82473234]),
                Tensor([1.0]),
                Tensor([1.0, -2.13808913, -2.88675129, -4.01188756]),
            ],
        ),
    ],
)
def test_local_frames_module(coeffs, irreps, pos, batch, atomic_numbers, expected):
    """Test the function local_frames_module with some test cases."""
    f_module = LocalFramesModule()
    trafo_coeffs = f_module.forward(coeffs, irreps, pos, atomic_numbers, batch)

    for i in range(len(coeffs)):
        assert trafo_coeffs[i].shape == coeffs[i].shape
        assert trafo_coeffs[i] == pytest.approx(expected[i])
        # assert torch.allclose(trafo_coeffs[i], expected[i])


@set_default_torch_dtype(torch.float64)
def test_local_frames_matrix_sparse(dummy_dataset_path, dummy_basis_info):
    """Test the LocalFramesTransformMatrixSparse class."""
    master_transform = MasterTransformation(
        "none", pre_transforms=[ToTorch(), AddAtomCooIndices()]
    )
    dataset = OFDataset.from_directory(
        dummy_dataset_path,
        basis_info=dummy_basis_info,
        add_irreps=True,
        transforms=master_transform,
    )
    get_local_frames_transform_matrix = LocalFramesTransformMatrixSparse()
    for sample in dataset:
        mat = get_local_frames_transform_matrix.sample_forward(sample)

        assert mat.shape == (sample.n_basis, sample.n_basis)
        transform_and_back = (mat.T @ mat).to_dense()
        torch.testing.assert_close(
            transform_and_back, torch.eye(sample.n_basis, dtype=mat.dtype, device=mat.device)
        )
    # Uncomment to (hopefully) see a nice block-diagonal matrix:
    # from matplotlib import pyplot as plt
    # plt.imshow(mat.to_dense())
    # plt.show()


@set_default_torch_dtype(torch.float64)
def test_local_frames_matrix_dense(
    dummy_dataset_path, dummy_basis_info, master_transform_to_torch
):
    """Test the LocalFramesTransformMatrixDense class."""
    dataset = OFDataset.from_directory(
        dummy_dataset_path,
        basis_info=dummy_basis_info,
        add_irreps=True,
        transforms=master_transform_to_torch,
    )
    get_local_frames_transfor_matrix = LocalFramesTransformMatrixDense()
    for sample in dataset:
        mat = get_local_frames_transfor_matrix.sample_forward(sample)

        assert mat.shape == (sample.n_basis, sample.n_basis)
        torch.testing.assert_close(
            mat.T @ mat, torch.eye(sample.n_basis, dtype=mat.dtype, device=mat.device)
        )
