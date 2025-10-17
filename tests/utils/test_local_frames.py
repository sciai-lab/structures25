import pytest
import torch
from e3nn.o3 import Irreps

from mldft.ml.models.components.local_frames_module import LocalBasisModule
from mldft.utils.local_frames import (
    get_rotation_matrix_from_basis,
    local_frames_from_positions,
    transform_coeffs_to_local,
    transform_local_coeffs_to_global,
)
from mldft.utils.utils import set_default_torch_dtype


@pytest.mark.parametrize(
    "pos, neighbor1_pos, neighbor2_pos, expected",
    [
        (
            torch.tensor([0.0, 0.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, 2.0, 0.0]),
            torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
        ),
        (
            torch.tensor([0.0, -1.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([-1.0, 2.0, 0.0]),
            torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [0.70710678, 0.70710678, 0.0],
                    [-0.70710678, 0.70710678, 0.0],
                ]
            ),
        ),
        (
            torch.tensor([-1.0, 0.0, 0.0]),
            torch.tensor([1.0, 1.0, 0.0]),
            torch.tensor([0.0, 2.0, 1.0]),
            torch.tensor(
                [
                    [0.26726127, -0.53452253, 0.80178374],
                    [0.89442718, 0.44721359, 0.0],
                    [-0.35856858, 0.71713716, 0.59761435],
                ]
            ),
        ),
    ],
)
def test_local_frames_from_positions(pos, neighbor1_pos, neighbor2_pos, expected):
    """Test the function local_frames_from_positions with some test cases."""
    torch.testing.assert_close(
        local_frames_from_positions(pos, neighbor1_pos, neighbor2_pos), expected
    )


@set_default_torch_dtype(torch.float64)
@pytest.mark.parametrize(
    "coeffs, irreps, basis",
    [
        (
            torch.tensor([1.0]),
            Irreps("0e"),
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
        ),
        (
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            Irreps("0e+1e"),
            torch.tensor(
                [
                    [0.89442718, 0.44721359, 0.0],
                    [-0.35856858, 0.71713716, 0.59761435],
                    [0.26726127, -0.53452253, 0.80178374],
                ]
            ),
        ),
        (
            torch.tensor([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            Irreps("0e+1e+2e"),
            torch.tensor(
                [
                    [0.89442718, 0.44721359, 0.0],
                    [-0.35856858, 0.71713716, 0.59761435],
                    [0.26726127, -0.53452253, 0.80178374],
                ]
            ),
        ),
    ],
)
def test_transform_coeffs_to_standard(coeffs, irreps, basis):
    """Test the function transform_coeffs_to_standard with some test cases."""
    # Test the transformation by first transforming to a local frame and then back to the standard frame
    trafo_coeffs = transform_coeffs_to_local(coeffs, irreps, basis)
    standard_coeffs = transform_local_coeffs_to_global(trafo_coeffs, irreps, basis)

    torch.testing.assert_close(standard_coeffs, coeffs)


@pytest.mark.parametrize(
    "pos, atomic_numbers, batch",
    [
        (
            torch.tensor(
                [
                    [-1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 2.0, 1.0],
                ]
            ),
            torch.tensor([2, 2, 2]),  # no hydrogen
            None,
        ),
    ],
)
def test_transform_coeffs_to_local(pos, atomic_numbers, batch):
    """Tests the transformation of the coefficients to the local frame."""
    local_bases = LocalBasisModule().forward(
        pos, atomic_numbers, batch
    )  # shape (n_atoms=3, n_basis_vectors=3, 3)

    irreps = Irreps("1e")

    for i in range(local_bases.shape[0]):
        # rotation matrix into the local frame
        rot_mat = get_rotation_matrix_from_basis(local_bases[i]).double()

        for j in range(local_bases.shape[1]):
            # jth basis vector in the local frame of atom i
            basis_vector = local_bases[i, j].double()

            # transform the basis vector into the local frame. this should be the jth unit vector
            basis_vector_in_local_frame = transform_coeffs_to_local(basis_vector, irreps, rot_mat)

            unit_vector = torch.zeros(3, dtype=torch.float64)
            unit_vector[j] = 1.0

            # see if the rot matrix works as expected
            assert torch.allclose(basis_vector_in_local_frame, unit_vector, atol=1e-5)

            assert torch.allclose(unit_vector, torch.matmul(rot_mat, basis_vector), atol=1e-5)
            assert torch.allclose(unit_vector, basis_vector @ rot_mat.T, atol=1e-5)
