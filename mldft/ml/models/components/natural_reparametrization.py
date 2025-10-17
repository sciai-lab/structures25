"""Natural reparametrization of the LCAB Ansatz for the electron density.

Taken as described in [M-OFDFT]_(https://doi.org/10.48550/arXiv.2309.16578), in the third paragraph of section 4.3 and appendix B.3.2
would yield the so called "canical" orthogonalization of the atomic orbital basis,

.. math::
    M = QΛ^{1/2}

where :math:`Q` is the matrix containing eigenvectors and :math:`Λ` the diagonal matrix of the
overlap matrix' :math:`W` eigenvalues .

However, the implementation in the authors' code uses the so called "symmetric" orthogonalization,
which corresponds to the symmetric orthogonalization of atomic orbitals
(cf. Löwdin 1970, https://www.sciencedirect.com/science/article/abs/pii/S0065327608603391),

.. math::
    M = QΛ^{1/2}Q^T.

While the former (canonical) is rather globally spread throughout the molecule and invariant with
respect to unitary transformations, the latter (symmetric) describes the original basis functions
best and is provably equivariant with respect to such transformations, e.g. local frame rotations.
"""

from typing import Literal

import numpy as np
import torch


def natural_reparametrization_matrices(
    overlap_matrix: np.ndarray,
    orthogonalization: Literal["symmetric", "canonical"] = "symmetric",
    do_sanity_checks: bool = False,
) -> (np.ndarray, np.ndarray):
    """Compute :math:`M`, the square-root of the overlap matrix :math:`W`, i.e. :math:`MM^T=W`, and
    its inverse :math:`M^{-1}`, used to transform coefficients and their gradients to the natural
    parametrization.

    Args:
        overlap_matrix: Overlap matrix (or matrices) of the atomic orbitals.
            Shape (..., n_ao, n_ao).
        orthogonalization: Type of orthogonalization to use. Either "symmetric" or "canonical".
            The symmetric version is the default.
        do_sanity_checks: Whether to perform sanity checks on the input and output.

    Returns:
        matrix :math:`M` and its inverse.
    """
    if do_sanity_checks:
        assert np.allclose(
            overlap_matrix, np.moveaxis(overlap_matrix, -1, -2)
        ), f"Overlap matrix must be symmetric. {overlap_matrix=}"

    # get eigenvalues and eigenvectors of the overlap matrix
    # using eigh because a) it's symmetric and b) eig does not guarantee that eigvec.T = eigvec^-1
    # c) its faster
    eigvals, eigvecs = np.linalg.eigh(overlap_matrix)

    if do_sanity_checks:
        # make sure eigenvalue decomposition worked as intended, i.e. W=QΛ(Q.T)
        # assert np.allclose(eigvecs @ np.diag(eigvals) @ eigvecs.T, overlap_matrix)
        assert np.allclose(
            np.einsum("...ij, ...j, ...kj", eigvecs, eigvals, eigvecs), overlap_matrix
        )

    eigvals_sqrt = np.sqrt(eigvals)
    eigvals_sqrt_inv = 1 / eigvals_sqrt

    if orthogonalization == "symmetric":  # M = Q @ Λ^0.5 @ Q.T, M^{-1} = Q @ Λ^{-0.5} @ Q.T
        m = np.einsum("...ij, ...j, ...kj -> ...ik", eigvecs, eigvals_sqrt, eigvecs)
        m_inv = np.einsum("...ij, ...j, ...kj -> ...ik", eigvecs, eigvals_sqrt_inv, eigvecs)

    elif orthogonalization == "canonical":  # M = Q @ Λ^0.5, M^{-1} = Λ^-{0.5} @ Q
        m = np.einsum("...ij, ...j -> ...ij", eigvecs, eigvals_sqrt)
        m_inv = np.einsum(" ...j ,...ij-> ...ji", eigvals_sqrt_inv, eigvecs)

    return m, m_inv


def natural_reparametrization_matrices_torch(
    overlap_matrix: torch.Tensor,
    orthogonalization: Literal["symmetric", "canonical"] = "symmetric",
    do_sanity_checks: bool = False,
) -> (torch.Tensor, torch.Tensor):
    """Compute :math:`M`, the square-root of the overlap matrix :math:`W`, i.e. :math:`MM^T=W`, and
    its inverse :math:`M^{-1}` as a torch.Tensor, used to transform coefficients and their
    gradients to the natural parametrization.

    Args:
        overlap_matrix: Overlap matrix (or matrices) of the atomic orbitals.
            Shape (..., n_ao, n_ao).
        orthogonalization: Type of orthogonalization to use. Either "symmetric" or "canonical".
            The symmetric version is the default.
        do_sanity_checks: Whether to perform sanity checks on the input and output.

    Returns:
        matrix :math:`M` and its inverse.
    """
    if do_sanity_checks:
        torch.testing.assert_close(
            overlap_matrix,
            overlap_matrix.transpose(-1, -2),
            atol=1e-5,
            rtol=1e-5,
            msg=f"Overlap matrix must be symmetric. {overlap_matrix=}",
        )

    # get eigenvalues and eigenvectors of the overlap matrix
    # using eigh because a) it's symmetric and b) eig does not guarantee that eigvec.T = eigvec^-1
    # c) its faster
    eigvals, eigvecs = torch.linalg.eigh(overlap_matrix)

    # make sure eigenvalue decomposition worked as intended, i.e. W=QΛ(Q.T)
    # assert torch.allclose(eigvecs @ torch.diag(eigvals) @ eigvecs.T, overlap_matrix)
    if do_sanity_checks:
        torch.testing.assert_close(
            torch.einsum("...ij, ...j, ...kj", eigvecs, eigvals, eigvecs),
            overlap_matrix,
            atol=1e-3,
            rtol=1e-4,
        )
    eigvals_sqrt = torch.sqrt(eigvals)

    eigvals_sqrt_inv = 1 / eigvals_sqrt

    if orthogonalization == "symmetric":  # M = Q @ Λ^0.5 @ Q.T, M^{-1} = Q @ Λ^{-0.5} @ Q.T
        m = torch.einsum("...ij, ...j, ...kj -> ...ik", eigvecs, eigvals_sqrt, eigvecs)
        m_inv = torch.einsum("...ij, ...j, ...kj -> ...ik", eigvecs, eigvals_sqrt_inv, eigvecs)

    elif orthogonalization == "canonical":  # M = Q @ Λ^0.5, M^{-1} = Λ^-{0.5} @ Q.T
        m = torch.einsum("...ij, ...j -> ...ij", eigvecs, eigvals_sqrt)
        m_inv = torch.einsum(" ...j ,...ij-> ...ji", eigvals_sqrt_inv, eigvecs)

    return m, m_inv
