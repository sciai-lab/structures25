# This file is using parts from https://github.com/atomicarchitects/equiformer_v2 licensed under the MIT License.

import os
from typing import Optional

import e3nn.o3._wigner as o3_wigner
import torch
from e3nn.o3 import Irreps
from torch import Tensor

# Borrowed from e3nn @ 0.4.0:
# https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L10
# _Jd is a list of tensors of shape (2l+1, 2l+1)
_Jd = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"), weights_only=True)


# Borrowed from e3nn @ 0.4.0:
# https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L37
#
# In 0.5.0, e3nn shifted to torch.matrix_exp which is significantly slower:
# https://github.com/e3nn/e3nn/blob/0.5.0/e3nn/o3/_wigner.py#L92
def wigner_D(l, alpha, beta, gamma):
    if not l < len(_Jd):
        raise NotImplementedError(
            f"wigner D maximum l implemented is {len(_Jd) - 1}, send us an email to ask for more"
        )

    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    J = _Jd[l].to(dtype=alpha.dtype, device=alpha.device)
    Xa = _z_rot_mat(alpha, l)
    Xb = _z_rot_mat(beta, l)
    Xc = _z_rot_mat(gamma, l)
    return Xa @ J @ Xb @ J @ Xc


def _z_rot_mat(angle, l):
    shape, device, dtype = angle.shape, angle.device, angle.dtype
    M = angle.new_zeros((*shape, 2 * l + 1, 2 * l + 1))
    inds = torch.arange(0, 2 * l + 1, 1, device=device)
    reversed_inds = torch.arange(2 * l, -1, -1, device=device)
    frequencies = torch.arange(l, -l - 1, -1, dtype=dtype, device=device)
    M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
    M[..., inds, inds] = torch.cos(frequencies * angle[..., None])
    return M


# monkey patch the wigner_D function from e3nn to the faster version
o3_wigner.wigner_D = wigner_D


def pyscf_to_e3nn_matrix(irreps: Irreps, dtype: torch.dtype = torch.float64) -> Tensor:
    """Calculates the transformation matrix which can be used to transform a coefficient vector in
    the pyscf convention to a coefficient vector in the e3nn convention.

    For details on the conventions, compare https://docs.e3nn.org/en/stable/guide/change_of_basis.html and
    https://pyscf.org/user/gto.html#ordering-of-basis-functions. The difference lies in the orientation
    of the spherical harmonics for l>1.

    A coefficient vector can be transformed by multiplying it with this matrix **on the left**.

    .. warning:
        As of now, the equivariance of the vector in the e3nn convention under e3nn transformations has not been
        checked explicitly

    Args:
        irreps: The irreps of the coefficient vector in the pyscf convention.
        dtype: The dtype of the matrix.

    Returns:
        torch.Tensor: The transformation matrix. Shape (irreps.dim, irreps.dim).
    """
    # pyscf uses a different convention than e3nn for l>1 (the wikipedia convention).
    # To alleviate the difference, we transform the l>1 vectors into the e3nn convention first.
    # To keep the l=1 vectors fixed, we replace them with scalars for this transformation:
    # Construct an irreps object where l=1 vectors are replaced by three l=0 scalars
    irreps_l_greater_1 = []
    for irrep in irreps:
        if irrep.ir.l == 1:
            irreps_l_greater_1.append(f"{3 * irrep.mul}x0e")
        else:
            irreps_l_greater_1.append(str(irrep))
    irreps_l_greater_1 = Irreps("+".join(irreps_l_greater_1))

    change_of_coord = torch.tensor(
        [
            # this specifies the change of basis yzx -> xyz
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=dtype,
    )

    return irreps_l_greater_1.D_from_matrix(change_of_coord)


def pyscf_to_e3nn_local_frames_matrix(basis: Tensor, irreps: Irreps) -> Tensor:
    """Calculates transformation matrix which can be used to transform a coefficient vector in the
    pyscf convention to a coefficient vector in local frames in the e3nn convention.

    A coefficient vector can be transformed by multiplying it with this matrix **on the left**.

    Args:
        basis: The basis vectors of the local frame. Shape (3, 3).
        irreps: The irreps of the coefficient vector in the pyscf convention.

    Returns:
        torch.Tensor: The transformation matrix. Shape (irreps.dim, irreps.dim).
    """

    # Ignore the four lines below, we do not need to do this here, as we refactored the local basis module:
    # Now, the second axis points towards the closest neighbor, alleviating the need for this permutation.
    # # Permute the basis to the order zyx.
    # # The reason is that e3nn uses y (the second axis) as the distinguished one, and we want this to correspond to the
    # # first axis in the local basis, which points towards the closest neighbor.
    # # basis = basis[[2, 0, 1], :]

    # Calculate the transformation matrix for transforming from the pyscf to the e3nn convention
    pyscf_to_e3nn_mat = pyscf_to_e3nn_matrix(irreps, dtype=basis.dtype)

    # Calculate the rotation matrix from the basis.
    rot_mat = get_rotation_matrix_from_basis(basis)

    # Calculate the transformation matrix for transforming to the local frames inside the e3nn convention.
    wigner_D_mat = irreps.D_from_matrix(rot_mat)

    return wigner_D_mat @ pyscf_to_e3nn_mat


def transform_coeffs_to_local(coeffs: Tensor, irreps: Irreps, basis: Tensor) -> Tensor:
    """Transforms the coefficients into the local frame.

    Args:
        coeffs (Tensor): Coefficients of the atom.
        irreps (Irreps): Irreps of the atom.
        basis (Tensor): Local basis of the atom.

    Returns:
        Tensor: The transformed coefficients in the local frame.
    """
    # multiply the coefficients with the Wigner matrices to transform them into the local frame
    return (
        pyscf_to_e3nn_local_frames_matrix(basis, irreps).to(coeffs.device, coeffs.dtype) @ coeffs
    )


def transform_local_coeffs_to_global(coeffs: Tensor, irreps: Irreps, basis: Tensor) -> Tensor:
    """Transforms the coefficients (in the e3nn convention) back into the standard basis.

    .. warning:
        The output of this function follows the pyscf convention,
        and is hence not equivariant under the e3nn convention.

    Args:
        coeffs (Tensor): Coefficients of the atom.
        irreps (Irreps): Irreps of the atom.
        basis (Tensor): Local basis of the atom.

    Returns:
        Tensor: The transformed coefficients into the local frame.
    """

    # multiply the coefficients with the Wigner matrices to transform them into the standard basis
    # here we use that the back rotation is the non transposed rotation matrix
    return (
        pyscf_to_e3nn_local_frames_matrix(basis, irreps).T.to(coeffs.device, coeffs.dtype) @ coeffs
    )


def get_rotation_matrix_from_basis(basis: Tensor) -> Tensor:
    """Returns the rotation matrix from the basis.

    Args:
        basis (Tensor): The basis of the local frame.

    Returns:
        Tensor: The rotation matrix to the new basis from the standard basis. Assumed to be multiplied as
        'vector @ rot_mat.T' or 'torch.matmul(rot_mat, vector)'.
    """

    # the rotation matrix is the just the basis matrix
    return basis


def local_frames_from_rel_positions(
    diff_vec_1: Tensor, diff_vec_2: Tensor, diff_vec_3: Optional[Tensor] = None
) -> Tensor:
    """Returns the local frames of the atoms in the molecule.

    In case the vectors are parallel, the second vector is changed to a random vector.

    .. warning:
        Different to how it is described in [M-OFDFT]_, the axis pointing towards the
        closest neighbor is the second axis. This is because the second axis is the
        distinguished axis in e3nn.
        In case of only wanting to put 2 pos in, make the third vector a 0 vector. Then the third axis is just chosen
        s.t. the local frame is right-handed.

    Args:
        diff_vec_1 (Tensor): The relative position of the first neighbor atom. Shape (..., 3)
        diff_vec_2 (Tensor): The relative position of the second neighbor atom. Shape (..., 3)
        diff_vec_3 (Optional[Tensor]): The relative position of the third neighbor atom. Shape (..., 3). If given, the basis system will be chosen such that the third axis is oriented towards the third neighbor.


    Returns:
        torch.Tensor: The local frames of the atoms in the molecule.

    Notes:
        In case the vectors are parallel, the second vector is changed to a random vector.
    """

    # calculate the basis vectors of the local frame with the cross products

    lengths = torch.norm(diff_vec_1, dim=-1)
    assert not torch.any(lengths == 0), f"zero distance, {lengths=}"

    out_shape = list(diff_vec_1.shape) + [3]

    # unsqueeze first two vectors to make batched operations possible
    basis_x = torch.div(diff_vec_1, torch.norm(diff_vec_1, dim=-1).unsqueeze(-1)).view(-1, 3)
    diff_vec_2 = diff_vec_2.view(-1, 3)

    assert not torch.any(torch.isnan(basis_x)), f"NaN in basis_x: {basis_x=}"

    # calculate the z basis vector with the cross product between x and the second direction
    # and normalize it
    cross_z = torch.cross(basis_x, diff_vec_2, dim=-1)
    # NOTE: if the vectors are parallel the norm of the cross product will be zero
    # thus the second vector then needs to be changed. For the time being we will simply
    # use a random second vector.
    # do this until we have a non zero cross product, likely single integration ...
    # but to be safe ...
    while torch.any(torch.norm(cross_z, dim=-1) == 0):
        # change second vector
        diff_vec_2 = torch.where(
            torch.norm(cross_z, dim=-1)[:, None] == 0,
            torch.randn_like(diff_vec_2),
            diff_vec_2,
        )
        cross_z = torch.cross(basis_x, diff_vec_2, dim=-1)
    basis_z = cross_z / torch.norm(cross_z, dim=-1).unsqueeze(-1)

    # calculate the y basis vector with the cross product between z and x
    # no need to normalize because x and z are normalized
    basis_y = torch.cross(basis_z, basis_x, dim=-1)

    # if the third vector is given, we can use it to orient the z axis in its direction
    if diff_vec_3 is not None:
        diff_vec_3 = diff_vec_3.view(-1, 3)
        basis_z = torch.where(
            torch.einsum("ij,ij->i", basis_z, diff_vec_3)[:, None] < 0, -basis_z, basis_z
        )

    # stack the vectors into one tensor and return it
    # return torch.stack((basis_x, basis_y, basis_z), dim=-2)
    return torch.stack((basis_z, basis_x, basis_y), dim=-2).view(out_shape)


def local_frames_from_positions(
    pos: Tensor,
    neighbor1_pos: Tensor,
    neighbor2_pos: Tensor,
    neighbor3_pos: Optional[Tensor] = None,
) -> Tensor:
    """Returns the local frames of the atoms in the molecule.
    Warning: This function does not use the third neighbor position.

    Args:
        pos (Tensor): The position of the atom. Shape (..., 3)
        neighbor1_pos (Tensor): The position of the first neighbor atom. Shape (..., 3)
        neighbor2_pos (Tensor): The position of the second neighbor atom. Shape (..., 3)
        neighbor3_pos (Optional[Tensor]): The position of the third neighbor atom. Shape (..., 3). If given, the basis system will be chosen such that the third axis is oriented towards the third neighbor.

    Returns:
        torch.Tensor: The local frames of the atoms in the molecule.
    """

    # calculate vectors which point from the start to the neighbors
    diff_vec_1 = neighbor1_pos - pos
    diff_vec_2 = neighbor2_pos - pos
    diff_vec_3 = neighbor3_pos - pos if neighbor3_pos is not None else None

    # calculate basis and return it
    return local_frames_from_rel_positions(diff_vec_1, diff_vec_2, diff_vec_3)
