r"""Functions to help with density fitting and value and gradient label calculation.

OFDFT uses different atomic basis functions for the density than KSDFT uses for the representation
of the orbitals. For the basis transformation the density coefficients in the OFDFT-basis are
fitted to the density defined by the orbitals, such that they minimize the error in the Hartree and
external Energy of the residual density.

The following definitions are used in the following derivations:

.. math::
    W_{\mu\nu} &= \langle\omega_\mu |\omega_\nu\rangle\\
    L_{\mu,\nu\gamma} &= \langle \omega_\mu |\eta_\nu\eta_\gamma\rangle\\
    D_{\alpha,\beta,\gamma,\delta}  &= \langle\eta^\alpha\eta^\beta|\eta^\gamma\eta^\delta\rangle\\
    \tilde{W}_{\mu\nu} &= (\omega_\mu |\omega_\nu)\\
    \tilde{L}_{\mu,\nu\gamma} &= (\omega_\mu |\eta_\nu\eta_\gamma)\\
    \tilde{D}_{\alpha,\beta,\gamma,\delta}  &= (\eta^\alpha\eta^\beta|\eta^\gamma\eta^\delta)\\
    S_{\alpha,\beta} &= \langle \eta^\alpha|\eta^\beta\rangle\\
    {v_{ext}}_\mu &= \int \omega_\mu (r) V_{ext} (r)dr\\
    {V_{ext}}_{\mu,\nu} &= \langle \eta_\mu | V_{ext} (r) | \eta_\nu \rangle\\
    \Gamma_{\alpha,\beta} &= \sum_i C_{\alpha,i}C^T_{i,\beta}
"""

from collections.abc import Callable
from typing import TypeVar

import numpy as np
import torch
from loguru import logger
from pyscf import dft, gto

from mldft.ofdft import basis_integrals
from mldft.ofdft.basis_integrals import (
    get_coulomb_tensor,
    get_coulomb_tensor_cached,
    get_normalization_vector,
    get_overlap_matrix,
    get_overlap_tensor,
)


def _check_data_format(results: dict, data_per_iteration: list[dict], mol: gto.Mole) -> None:
    r"""Checks if the data provided by the .chk file contains all the parameters required for the
    density fitting and that the provided molecule object fits this data dimension wise.

    Args:
        results: The dict containing the parameters of the ksdft calculation
        data_per_iteration: List of dicts containing the data of every individual scf-iteration
        mol: The molecule object providing information about the atoms and the basis set

    Raises:
        KeyError: if one of the required Keys is not inside of the result dict
        ValueError: if the Number of basis functions in Data and Molecule is different
    """
    result_keys = (
        "converged",
        "total_energy",
        "occupation_numbers_orbitals",
        "molecular_coeffs_orbitals",
        "max_cycle",
        "name_xc_functional",
        "init_guess",
        "convergence_tolerance",
        "diis_start_cycle",
        "diis_space",
        "diis_method",
        "grid_level",
    )
    for key in result_keys:
        if key not in results:
            raise KeyError(
                f"Key {key} not found in results in .chk file. Label calculation not possible"
            )
    iteration_data_keys = (
        "diis_coefficients",
        "occupation_numbers_orbitals",
        "molecular_coeffs_orbitals",
        "total_energy",
        "coulomb_energy",
        "exchange_correlation_energy",
    )
    for key in iteration_data_keys:
        for i, data in enumerate(data_per_iteration):
            if key not in data:
                raise KeyError(
                    f"Key {key} not found in data of iteration {i} in .chk file. Label calculation not possible"
                )
    n_orbital_basis_functions = results["molecular_coeffs_orbitals"].shape[0]
    if not n_orbital_basis_functions == mol.nao_nr():
        raise ValueError("Molecule has different number of basis functions as KSDFT_Results!")


def get_density_fitting_function(
    method_name: str,
    mol_orbital_basis: gto.Mole,
    mol_density_basis: gto.Mole,
    W_coulomb: np.ndarray,
    v_external_C: np.ndarray,
    v_external_p: np.ndarray,
    max_memory: int | float | None = 4000,
) -> Callable[[np.ndarray], np.ndarray]:
    r"""Return a Python function density_fitting(gamma: np.ndarray) -> np.ndarray which transforms
    the density coefficients in the orbital basis into the coefficients in the orbital free basis
    according to a specified method.

    .. math::

        p^i = \mathbf{P}^i_{jk}\Gamma^{jk}

    Args:
        method_name: The name of the method used to calculate the density fitting
        mol_orbital_basis: The molecule in the orbital basis
        mol_density_basis: The molecule in the density basis
        W_coulomb: 2-center coulomb matrix
        v_external_p: 1 center external potential vector of the of-coefficients
        v_external_C: 1 center external potential matrix of the ks-coefficients
        max_memory: The maximum memory to use per process in MB.


    The tensor is based on the method to choose the coefficients such that they optimize a target.
    """
    # Currently only for hartree+external_mofdft the 3 integral tensor is used memory efficiently, otherwise we compute
    # the full tensor, in hartree+external at least the sum is computed memory efficiently
    if method_name != "hartree+external_mofdft":
        L_coulomb = get_coulomb_tensor_cached(mol_density_basis, mol_orbital_basis)

    if method_name == "hartree+external":

        def density_fitting(gamma: np.ndarray):
            return density_fitting_hartree_external(
                W_coulomb,
                L_coulomb,
                v_external_p,
                v_external_C,
                gamma,
                max_memory=max_memory,
            )

    elif method_name == "hartree":

        def density_fitting(gamma: np.ndarray):
            return density_fitting_hartree(W_coulomb, L_coulomb, gamma)

    elif method_name == "hartree+external_mofdft":

        def density_fitting(gamma: np.ndarray):
            return density_fitting_mofdft(
                mol_orbital_basis,
                mol_density_basis,
                W_coulomb,
                v_external_p,
                v_external_C,
                gamma,
                max_memory,
            )

    elif method_name == "hartree+external_mofdft_fixed_density":
        basis_integrals = get_normalization_vector(mol_density_basis)
        # S_overlap = get_overlap_matrix(mol_orbital_basis)
        n_electrons = np.asarray(mol_orbital_basis.nelectron)

        def density_fitting(gamma: np.ndarray):
            return density_fitting_mofdft_fixed_density(
                W_coulomb,
                L_coulomb,
                v_external_p,
                v_external_C,
                basis_integrals,
                n_electrons,
                gamma,
            )

    elif method_name == "hartree+external_mofdft_enforced_density":
        basis_integrals = get_normalization_vector(mol_density_basis)
        S_overlap = get_overlap_matrix(mol_orbital_basis)

        def density_fitting(gamma: np.ndarray):
            return density_fitting_mofdft_enforced_density(
                W_coulomb,
                L_coulomb,
                v_external_p,
                v_external_C,
                basis_integrals,
                S_overlap,
                gamma,
            )

    elif method_name == "overlap":
        W_overlap = get_overlap_matrix(mol_density_basis)
        L_overlap = get_overlap_tensor(mol_density_basis, mol_orbital_basis)

        def density_fitting(gamma: np.ndarray):
            return density_fitting_hartree(W_overlap, L_overlap, gamma)

    elif method_name == "hartree+external_fixed_density":
        basis_integrals = get_normalization_vector(mol_density_basis)
        # S_overlap = get_overlap_matrix(mol_orbital_basis)
        n_electrons = np.asarray(mol_orbital_basis.nelectron)

        def density_fitting(gamma: np.ndarray):
            return density_fitting_hartree_external_fixed_density(
                W_coulomb,
                L_coulomb,
                v_external_p,
                v_external_C,
                basis_integrals,
                n_electrons,
                gamma,
            )

    elif method_name == "hartree_fixed_density_external":
        basis_integrals = get_normalization_vector(mol_density_basis)
        # S_overlap = get_overlap_matrix(mol_orbital_basis)
        n_electrons = np.asarray(mol_orbital_basis.nelectron)

        def density_fitting(gamma: np.ndarray):
            return density_fitting_hartree_fixed_density_external(
                W_coulomb,
                L_coulomb,
                v_external_p,
                v_external_C,
                basis_integrals,
                n_electrons,
                gamma,
            )

    elif method_name == "overlap+external_fixed_density":
        W_overlap = get_overlap_matrix(mol_density_basis)
        L_overlap = get_overlap_tensor(mol_density_basis, mol_orbital_basis)
        basis_integrals = get_normalization_vector(mol_density_basis)
        # S_overlap = get_overlap_matrix(mol_orbital_basis)
        n_electrons = np.asarray(mol_orbital_basis.nelectron)

        def density_fitting(gamma: np.ndarray):
            return density_fitting_hartree_external_fixed_density(
                W_overlap,
                L_overlap,
                v_external_p,
                v_external_C,
                basis_integrals,
                n_electrons,
                gamma,
            )

    elif method_name == "overlap_fixed_density_external":
        W_overlap = get_overlap_matrix(mol_density_basis)
        L_overlap = get_overlap_tensor(mol_density_basis, mol_orbital_basis)
        basis_integrals = get_normalization_vector(mol_density_basis)
        # S_overlap = get_overlap_matrix(mol_orbital_basis)
        n_electrons = np.asarray(mol_orbital_basis.nelectron)

        def density_fitting(gamma: np.ndarray):
            return density_fitting_hartree_fixed_density_external(
                W_overlap,
                L_overlap,
                v_external_p,
                v_external_C,
                basis_integrals,
                n_electrons,
                gamma,
            )

    else:
        raise NotImplementedError(f"Unknown density fitting method: {method_name}")
    return density_fitting


def contract_coulomb_tensor(
    mol_orbital_basis: gto.Mole,
    mol_density_basis: gto.Mole,
    gamma: np.ndarray | torch.Tensor,
    max_memory: int | float | None = 4000,
) -> torch.Tensor:
    """Computes the contraction of the 3-center coulomb tensor with the density coefficients in the
    orbital basis.

    Args:
        mol_orbital_basis: The molecule in the orbital basis
        mol_density_basis: The molecule in the density basis
        gamma: The density coefficients in the orbital basis
        max_memory: The maximum memory to use per process in MB.

    Returns:
        The contracted coulomb tensor as a numpy array.
    """
    if isinstance(gamma, np.ndarray):
        gamma = torch.from_numpy(gamma)
    coulomb_tensor_size = 2 * 8 * mol_orbital_basis.nao**2 * mol_density_basis.nao / 1048576
    if coulomb_tensor_size > max_memory:
        b = []
        for i in range(mol_density_basis.nbas):
            shls_slice = (
                0,
                mol_orbital_basis.nbas,
                0,
                mol_orbital_basis.nbas,
                mol_orbital_basis.nbas + i,
                mol_orbital_basis.nbas + i + 1,
            )
            L_coulomb_slice = get_coulomb_tensor(
                mol_density_basis, mol_orbital_basis, shls_slice=shls_slice
            )
            L_coloumb_torch = torch.from_numpy(L_coulomb_slice)
            b.append((L_coloumb_torch * gamma).sum(dim=(1, 2)))
        b = torch.cat(b)
    else:
        L_coulomb = get_coulomb_tensor_cached(mol_density_basis, mol_orbital_basis)
        L_coulomb = torch.from_numpy(L_coulomb)
        b = torch.einsum("ijk,jk->i", L_coulomb, gamma)
    return b


def density_fitting_hartree(
    W_coulomb: np.ndarray, L_coulomb: np.ndarray, gamma: np.ndarray
) -> np.ndarray:
    r""" Constructs the (n_auxmol) tensor which describes a the of-coefficients.
    The target to optimize is the Hartree energy of the residual density:

    .. math::

        \begin{aligned}
        \mathcal{L}(\mathbf{p}) &= \mathbf{p} \tilde{W} \mathbf{p} - 2 \mathbf{p}\bar {\tilde L} \bar\Gamma + \bar\Gamma \tilde{\mathbf{D}}\bar\Gamma
        \end{aligned}

    This is solved by assuming that :math:`\tilde{W}` is invertible:

    .. math::

        \begin{aligned}
        \partial_{\mathbf p}\mathcal L&= 2\tilde{W} \mathbf{p}- 2 \bar {\tilde L} \bar\Gamma=0\\
        \mathbf{p}&=\tilde{W}^{-1}\bar {\tilde L} \bar\Gamma\\
        &=\bar{\mathbf{P}} \bar\Gamma
        \end{aligned}

    Args:
        L_overlap: 3-center coulomb matrix
        W_overlap: 2-center coulomb matrix
        gamma: coefficients of the density in the orbital basis

    Returns:
        P: the 3-index tensor which maps Gamma to p
    Notes:
        Corresponds to ´´df_coeff´´¸in the MOFDFT implementation. But is probably not used by them.
        This function is also used to compute the map if one wants to minimize the L2 - norm of the residual density.
        For this W_coulomb and L_coulomb have to be replaced with W_overlap and L_overlap.
    """
    a = W_coulomb
    b = np.einsum("ijk,jk->i", L_coulomb, gamma, optimize=True)
    coeff = np.linalg.lstsq(a, b, rcond=None)[0]
    return coeff


def density_fitting_hartree_external(
    W_coulomb: np.ndarray,
    L_coulomb: np.ndarray,
    v_external_p: np.ndarray,
    v_external_C: np.ndarray,
    gamma: np.ndarray,
    max_memory: int | float | None = 4000,
) -> np.ndarray:
    r""" Constructs the (n_auxmol) tensor which describes a the of-coefficients.
    The target to optimize is the sum of Hartree energy and external energy of the residual density:

    .. math::

        \begin{aligned}
        \mathcal{L}(\mathbf{p}) &= \mathbf{p} \tilde{W} \mathbf{p} - 2 \mathbf{p}\bar {\tilde L} \bar\Gamma + \bar\Gamma \tilde{\mathbf{D}}\bar\Gamma + (\mathbf{p}\mathbf{v}_{ext}-\bar\Gamma \bar{V}_{ext})^2
        \end{aligned}

    This is solved by assuming that :math:`A=\tilde{W}+\mathbf{v}_{ext}\mathbf{v}_{ext}^T` is invertible:

    .. math::

        \begin{aligned}
        \partial_{\mathbf p}\mathcal L&= 2\tilde{W} \mathbf{p}- 2 \bar {\tilde L} \bar\Gamma + 2\mathbf{v}_{ext}\mathbf{v}_{ext}^T\mathbf{p} - 2\mathbf{v}_{ext}\bar\Gamma \bar{V}_{ext}\\
        &= 2 A \mathbf{p}- 2 \bar {\tilde L} \bar\Gamma - 2\mathbf{v}_{ext}\bar\Gamma \bar{V}_{ext}=0\\
        \mathbf{p}&=A^{-1}\left(\bar {\tilde L}+\mathbf{v}_{ext}\bar{V}_{ext}\right) \bar\Gamma\\
        &=\bar{\mathbf{P}} \bar\Gamma
        \end{aligned}

    Args:
        W_coulomb: 2-center coulomb matrix of the of-coefficients
        L_coulomb: 3-center coulomb matrix of the overlap of ks- and of-coefficients
        v_external_p: 1 center external potential vector of the of-coefficients
        v_external_C: 1 center external potential matrix of the ks-coefficients
        gamma: coefficients of the density in the orbital basis
        max_memory: The maximum memory to use per process in MB.

    Returns:
        P: the 3-index tensor which maps Gamma to p
    Notes:
        The energy lagragian is mentioned in the [M-OFDFT]_ paper but the equation they derive does not minimize it.
        This function depicts the correct solution to the minimization problem.
    """
    A = W_coulomb + np.outer(v_external_p, v_external_p)
    # If the L_coulomb matrix is too large, don't use einsum as it doubles the memory usage
    if 2 * L_coulomb.size * 8 / 1048576 > max_memory:
        L_effective = torch.zeros(L_coulomb.shape[0])
        L_coloumb_torch = torch.from_numpy(L_coulomb)
        gamma_torch = torch.from_numpy(gamma)
        for i in range(L_effective.shape[0]):
            L_effective[i] = torch.sum(L_coloumb_torch[i] * gamma_torch)
        L_effective = L_effective.numpy()
    else:
        # numpy einsum does not multithread here so use torch einsum
        L_effective = torch.einsum(
            "ijk,jk->i", torch.from_numpy(L_coulomb), torch.from_numpy(gamma)
        ).numpy()
    L_effective += v_external_p * np.einsum("ij,ij->", v_external_C, gamma, optimize=True)
    # This is faster for large molecules but seems to be less symmetric/accurate for quantities of small molecules
    # It was used for the qmugs generation
    # coeff = scipy.linalg.solve(A, L_effective, assume_a="sym")
    coeff = np.linalg.lstsq(A, L_effective, rcond=None)[0]
    return coeff


def density_fitting_mofdft(
    mol_orbital_basis,
    mol_density_basis,
    W_coulomb: np.ndarray,
    v_external_p: np.ndarray,
    v_external_C: np.ndarray,
    gamma: np.ndarray,
    max_memory: int | float | None = 4000,
) -> np.ndarray:
    r"""Constructs the (n_auxmol) tensor which describes a the of-coefficients.
    The target to optimize is mentioned as the sum of the Hartree energy of the residual density and the external energy:

    .. math::

        \begin{aligned}
        \mathcal{L}(\mathbf{p}) &= \mathbf{p} \tilde{W} \mathbf{p} - 2 \mathbf{p}\bar {\tilde L} \bar\Gamma + \bar\Gamma \tilde{\mathbf{D}}\bar\Gamma + (\mathbf{p}\mathbf{v}_{ext}-\bar\Gamma \bar{V}_{ext})^2
        \end{aligned}

    But the following equation does not necessarily minimizes this Energy function.

    .. math::

        \left(\begin{array}{c}\tilde{W}\\v_{ext}^T\end{array}\right) \mathbf{p} =  \left(\begin{array}{c}\tilde{L} \bar{\Gamma} \\ \bar{\Gamma}\bar{V}_{ext}\end{array}\right)

    It is solved using least squares methods.
    Args:
        mol_orbital_basis: The molecule in the orbital basis
        mol_density_basis: The molecule in the density basis
        W_coulomb: 2-center coulomb matrix of the of-coefficients
        v_external_p: 1 center external potential vector of the of-coefficients
        v_external_C: 1 center external potential matrix of the ks-coefficients
        gamma: coefficients of the density in the orbital basis
        max_memory: The maximum memory to use per process in MB.

    Returns:
        P: the 3-index tensor which maps Gamma to p
    Notes:
        The energy Lagragian is mentioned in the [M-OFDFT]_ paper but the equation derived does not minimize it.
        Called ``df_coeff_jext`` in the MOFDFT ìmplementation. This is the method mentioned [M-OFDFT]_ mention
        to use in their implementation.
    """
    return density_fitting_mofdft_torch(
        mol_orbital_basis,
        mol_density_basis,
        W_coulomb,
        v_external_p,
        v_external_C,
        torch.from_numpy(gamma),
        max_memory,
    ).numpy()


def density_fitting_mofdft_torch(
    mol_orbital_basis: gto.Mole,
    mol_density_basis: gto.Mole,
    W_coulomb: np.ndarray,
    v_external_p: np.ndarray,
    v_external_C: np.ndarray,
    gamma: torch.Tensor,
    max_memory: int | float | None = 4000,
) -> torch.Tensor:
    """Same as :py:func:`density_fitting_mofdft`, but using torch (backpropagatable)

    Args:
        W_coulomb: 2-center coulomb matrix of the of-coefficients
        contracted_coulomb_tensor: The contracted coulomb tensor
        v_external_p: 1 center external potential vector of the of-coefficients
        v_external_C: 1 center external potential matrix of the ks-coefficients
        gamma: coefficients of the density in the orbital basis
        max_memory: The maximum memory to use per process in MB.
    """
    tensor_type = {"dtype": gamma.dtype, "device": gamma.device}
    a = np.concatenate([W_coulomb, v_external_p[None]], axis=0)
    a = torch.as_tensor(a, **tensor_type)
    v_external_C = torch.as_tensor(v_external_C, **tensor_type)
    logger.trace("Computing energy terms.")
    b0 = contract_coulomb_tensor(mol_orbital_basis, mol_density_basis, gamma, max_memory)
    b1 = torch.einsum("ij,ij->", v_external_C, gamma)
    b = torch.concatenate((b0, b1[None]), dim=0)
    logger.trace("Solving linear system.")
    coeff = torch.linalg.lstsq(a, b, rcond=None)[0]
    return coeff


def density_fitting_mofdft_enforced_density(
    W_coulomb: np.ndarray,
    L_coulomb: np.ndarray,
    v_external_p: np.ndarray,
    v_external_C: np.ndarray,
    basis_integrals: np.ndarray,
    overlap_matrix: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    r"""Constructs the (n_auxmol) tensor which describes the of-coefficients.

    The target to optimize is mentioned as the sum of the Hartree energy of the residual density and the external energy modified and also to enforce density conversation.
    The minimized lagrangian is defined as follows:

    .. math::
        \begin{aligned}
        \mathcal{L}(\mathbf{p}) &= \left\lVert\left(\begin{array}{c}\tilde{W}\\v_{ext}^T\end{array}\right) \mathbf{p} - \left(\begin{array}{c}\tilde{L} \bar{\Gamma} \\ \bar{\Gamma}\bar{V}_{ext}\end{array}\right)\right\rVert^2 + \lambda (\mathbf{w}\mathbf{p}-N)
        \end{aligned}


    This is solved by the following equation and least squares methods.

    .. math::
        \begin{aligned}
        \mathbf{\tilde{p}} &= \text{argmin} \left\lVert\left(\begin{array}{c}\tilde{W}\\v_{ext}^T\end{array}\right) \mathbf{p} - \left(\begin{array}{c}\tilde{L} \bar{\Gamma} \\ \bar{\Gamma}\bar{V}_{ext}\end{array}\right)\right\rVert^2
        \mathbf{M} &= \left(\begin{array}{c}\tilde{W}\\v_{ext}^T\end{array}\right)^T\left(\begin{array}{c}\tilde{W}\\v_{ext}^T\end{array}\right)
        \mathbf{p} &= \mathbf{\tilde{p}} - \frac{\mathbf{w\tilde{p}}-N}{\mathbf{w}\mathbf{M}^{-1}\mathbf{w}}\mathbf{M}^{-1}\mathbf{w}
        \end{aligned}

    It is solved using least squares methods.

    Args:
        L_overlap: 3-center coulomb matrix of the overlap of ks- and of-coefficients
        W_overlap: 2-center coulomb matrix of the of-coefficients
        v_external_p: 1 center external potential vector of the of-coefficients
        v_external_C: 1 center external potential matrix of the ks-coefficients
        gamma: coefficients of the density in the orbital basis

    Returns:
        P: the 3-index tensor which maps Gamma to p
    Notes:
        This is a modification of the mofdft version of density fitting which enforces the conservation of the electron number.
    """
    a = np.concatenate([W_coulomb, v_external_p[None]], axis=0)
    b0 = np.einsum("ijk,jk->i", L_coulomb, gamma, optimize=True)
    b1 = np.einsum("ij,ij->", v_external_C, gamma, optimize=True)
    N = np.einsum("ij,ij->", overlap_matrix, gamma, optimize=True)
    b = np.concatenate([b0, b1[None]], axis=0)

    W = np.einsum("ji,jl->il", a, a)
    W_inv_w = np.linalg.lstsq(W, basis_integrals, rcond=None)[0]
    w_W_inv_w = basis_integrals @ W_inv_w
    coeff = np.linalg.lstsq(a, b, rcond=None)[0]

    w_tilde_p = basis_integrals @ coeff
    lagrange_multiplier = (w_tilde_p - N) / w_W_inv_w

    return coeff - lagrange_multiplier * W_inv_w


def density_fitting_mofdft_fixed_density(
    W_coulomb: np.ndarray,
    L_coulomb: np.ndarray,
    v_external_p: np.ndarray,
    v_external_C: np.ndarray,
    basis_integrals: np.ndarray,
    n_electrons: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    r"""Constructs the (n_auxmol) tensor which describes a the of-coefficients.
    The target to optimize is mentioned as the sum of the Hartree energy of the residual density and the external energy as well as the differences in L1 norm of the density:

    .. math::

        \begin{aligned}
        \mathcal{L}(\mathbf{p}) &= \mathbf{p} \tilde{W} \mathbf{p} - 2 \mathbf{p}\bar {\tilde L} \bar\Gamma + \bar\Gamma \tilde{\mathbf{D}}\bar\Gamma + (\mathbf{p}\mathbf{v}_{ext}-\bar\Gamma \bar{V}_{ext})^2 + (\mathbf{p}\mathbf{w}-\bar\Gamma \bar{S})^2
        \end{aligned}

    But the following equation does not necessarily minimizes this Energy function.

    .. math::

        \left(\begin{array}{c}\tilde{W}\\\mathbf{v}_{ext}^T\\\mathbf{w}^T\end{array}\right) \mathbf{p} =  \left(\begin{array}{c}\tilde{L} \bar{\Gamma} \\ \bar{\Gamma}\bar{V}_{ext}\\\bar S\bar \Gamma\end{array}\right)

    It is solved using least squares methods.
    Args:
        L_overlap: 3-center coulomb matrix of the overlap of ks- and of-coefficients
        W_overlap: 2-center coulomb matrix of the of-coefficients
        v_external_p: 1 center external potential vector of the of-coefficients
        v_external_C: 1 center external potential matrix of the ks-coefficients
        basis_integrals: 1 center integrals over the basis functions of the of basis
        overlap_matrix: 1 center intergrals over products of the basis functions of the ks-basis
        gamma: coefficients of the density in the orbital basis

    Returns:
        P: the 3-index tensor which maps Gamma to p
    Notes:
        The energy Lagragian is not mentioned in [M-OFDFT]_ , but it is implemented in the MOFDFT Github project.
        Called ``get_rho_coeff_jextnelec_fit`` in the MOFDFT ìmplementation.
    """
    int_1c1e_nuc = v_external_p
    a = np.concatenate([W_coulomb, int_1c1e_nuc[None], basis_integrals[None]], axis=0)
    b0 = np.einsum("ijk,jk->i", L_coulomb, gamma, optimize=True)
    b1 = np.einsum("ij,ij->", v_external_C, gamma, optimize=True)
    b2 = n_electrons  # np.einsum("ij,ij->", overlap_matrix, gamma, optimize=True)
    b = np.concatenate([b0, b1[None], b2[None]], axis=0)
    coeff = np.linalg.lstsq(a, b, rcond=None)[0]
    return coeff


def density_fitting_hartree_external_fixed_density(
    W_coulomb: np.ndarray,
    L_coulomb: np.ndarray,
    v_external_p: np.ndarray,
    v_external_C: np.ndarray,
    basis_integrals: np.ndarray,
    n_electrons: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    r""" Constructs the (n_auxmol) tensor which describes a the of-coefficients.
     The target to optimize is the sum of Hartree energy and external energy of the residual density,
      the L1 norm of the density and the external energy are enforced to stay constant after the mapping:


    .. math::

        \begin{aligned}
        \mathcal{L}(\mathbf{p},\mu) &= \mathbf{p} \tilde{W} \mathbf{p} - 2 \mathbf{p}\bar {\tilde L} \bar\Gamma + (\mathbf{p}\mathbf{v}_{ext}-\bar\Gamma \bar{V}_{ext})^2+\mu(\mathbf{p}\mathbf{w}-\bar\Gamma\bar S)\\
        \partial_{\mathbf p}\mathcal L&= 2\tilde{W} \mathbf{p}- 2 \bar {\tilde L} \bar\Gamma + 2\mathbf{v}_{ext}(\mathbf{p}\mathbf{v}_{ext}-\bar\Gamma \bar{V}_{ext})+\mu \mathbf{w}\\
        &= 2(\tilde{W}+\mathbf{v}_{ext}\mathbf{v}_{ext}^T) \mathbf{p}- 2 \bar {\tilde L} \bar\Gamma + 2\mathbf{v}_{ext}\bar\Gamma \bar{V}_{ext}+\mu \mathbf{w}= 0\\
        \partial_\mu\mathcal{L}&= (\mathbf{p}\mathbf{w}-\bar\Gamma\bar S)=0\\
        \end{aligned}

    If we assume that :math:`A=\tilde{W}+\mathbf{v}_{ext}\mathbf{v}_{ext}^T` is invertible

    .. math::

        \begin{aligned}
        \mathbf{w}A^{-1}\partial_{\mathbf p}\mathcal L&= 2 \mathbf{w}\mathbf{p}-2 \mathbf{w}A^{-1}\bar {\tilde L} \bar\Gamma-2\mathbf{w}A^{-1}\mathbf{v}_{ext}\bar\Gamma \bar{V}_{ext} + \mu \mathbf{w}A^{-1}\mathbf{w}\\
        &=2\bar\Gamma\bar S -2 \mathbf{w}A^{-1}\bar {\tilde L} \bar\Gamma-2\mathbf{w}A^{-1}\mathbf{v}_{ext}\bar\Gamma \bar{V}_{ext} + \mu \mathbf{w}A^{-1}\mathbf{w}=0\\
        \mu &= 2\frac{-\bar\Gamma\bar S + \mathbf{w}A^{-1}\bar {\tilde L} \bar\Gamma+\mathbf{w}A^{-1}\mathbf{v}_{ext}\bar\Gamma \bar{V}_{ext}}{\mathbf{w}A^{-1}\mathbf{w}}\\
        &=2\frac{\mathbf{w}A^{-1}\bar {\tilde L} +\mathbf{w}A^{-1}\mathbf{v}_{ext} \bar{V}_{ext}-\bar S}{\mathbf{w}A^{-1}\mathbf{w}}\bar\Gamma\\
        \partial_{\mathbf p}\mathcal L&= 2A\mathbf{p}- 2 \bar {\tilde L} \bar\Gamma - 2\mathbf{v}_{ext}\bar\Gamma \bar{V}_{ext}+\mu \mathbf{w}=0\\
        \Leftrightarrow\mathbf p&=A^{-1}\bar {\tilde L} \bar\Gamma + A^{-1}\mathbf{v}_{ext}\bar\Gamma \bar{V}_{ext}-\frac{1}{2}\mu A^{-1}\mathbf{w}\\
        &=A^{-1}\bar {\tilde L} \bar\Gamma + A^{-1}\mathbf{v}_{ext}\bar\Gamma \bar{V}_{ext}- A^{-1}\mathbf{w}\frac{\mathbf{w}A^{-1}\bar {\tilde L} +\mathbf{w}A^{-1}\mathbf{v}_{ext} \bar{V}_{ext}-\bar S}{\mathbf{w}A^{-1}\mathbf{w}}\bar\Gamma\\
        &=A^{-1}\left(\bar {\tilde L} + \mathbf{v}_{ext} \bar{V}_{ext}- \mathbf{w}\frac{\mathbf{w}A^{-1}\bar {\tilde L} +\mathbf{w}A^{-1}\mathbf{v}_{ext} \bar{V}_{ext}-\bar S}{\mathbf{w}A^{-1}\mathbf{w}}\right)\bar\Gamma\\
        &= \bar {\mathbf{P}}\bar\Gamma
        \end{aligned}

    Where :math:`\bar{\mathbf{P}}` is a three index tensor only dependent on the geometry of the molecule.

    Args:
        W_coulomb: 2-center overlap matrix
        L_coulomb: 3-center overlap matrix
        basis_integrals: integrals over the basis functions
        v_external_p: the density coefficients of the external potential
        v_external_C: the orbital coefficients of the external potential
        overlap_matrix: 2-center overlap matrix
        gamma: coefficients of the density in the orbital basis


    Returns:
        :math:`\bar{\mathbf{P}}`: the density coefficients in the new space
    """
    A = W_coulomb + v_external_p[:, None] @ v_external_p[None, :]
    L_gamma = np.einsum("ijk,jk->i", L_coulomb, gamma, optimize=True)
    V_ext_gamma = np.einsum("ij,ij->", v_external_C, gamma, optimize=True)
    overlap_gamma = n_electrons  # np.einsum("ij,ij->", overlap_matrix, gamma, optimize=True)
    A_inv_w = np.linalg.lstsq(A, basis_integrals, rcond=None)[0]
    w_A_inv_w = basis_integrals @ A_inv_w
    w_A_inv_L = A_inv_w @ L_gamma
    w_A_inv_v = A_inv_w @ v_external_p
    matrix = (w_A_inv_L + w_A_inv_v * V_ext_gamma - overlap_gamma) / w_A_inv_w
    right_side_eq = L_gamma + v_external_p * V_ext_gamma - basis_integrals * matrix
    coeffs = np.linalg.lstsq(A, right_side_eq, rcond=None)[0]
    return coeffs


def density_fitting_hartree_fixed_density_external(
    W_coulomb: np.ndarray,
    L_coulomb: np.ndarray,
    v_external_p: np.ndarray,
    v_external_C: np.ndarray,
    basis_integrals: np.ndarray,
    n_electrons: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    r""" Constructs the (n_auxmol) tensor which describes a the of-coefficients.
    The target to optimize is the Hartree energy of the residual density,
    the L1 norm of the density and the external energy are enforced to stay constant after the mapping:

    .. math::

        \begin{aligned}
        \mathcal{L}(\mathbf{p},\mu,\nu) &= \mathbf{p} \tilde{W} \mathbf{p} - 2 \mathbf{p}\bar {\tilde L} \bar\Gamma + \nu(\mathbf{p}\mathbf{v}_{ext}-\bar\Gamma \bar{V}_{ext})+\mu(\mathbf{p}\mathbf{w}-\bar\Gamma\bar S)\\
        \partial_{\mathbf p}\mathcal L&= 2\tilde{W} \mathbf{p}- 2 \bar {\tilde L} \bar\Gamma + \nu\mathbf{v}_{ext}+\mu \mathbf{w}= 0\\
        \partial_\mu\mathcal{L}&= (\mathbf{p}\mathbf{w}-\bar\Gamma\bar S)=0\\
        \partial_\nu\mathcal{L}&= (\mathbf{p}\mathbf{v}_{ext}-\bar\Gamma \bar{V}_{ext})=0\\
        \end{aligned}

    If we assume that :math:`A=\tilde{W}` is invertible

    .. math::
        \begin{aligned}
        \mathbf{w}\tilde W^{-1}\partial_{\mathbf p}\mathcal L&= 2 \mathbf{w}\mathbf{p}-2 \mathbf{w}\tilde W^{-1}\bar {\tilde L} \bar\Gamma+\nu\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext} + \mu \mathbf{w}\tilde W^{-1}\mathbf{w}\\
        &=2\bar\Gamma\bar S -2 \mathbf{w}\tilde W^{-1}\bar {\tilde L} \bar\Gamma+\nu\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext} + \mu \mathbf{w}\tilde W^{-1}\mathbf{w}=0\\
        \mu &= 2\frac{\bar\Gamma\bar S - \mathbf{w}\tilde W^{-1}\bar {\tilde L} \bar\Gamma}{\mathbf{w}\tilde W^{-1}\mathbf{w}}+\nu\frac{\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext}}{\mathbf{w}\tilde W^{-1}\mathbf{w}}\\
        \mathbf{v}_{ext}\tilde W^{-1}\partial_{\mathbf p}\mathcal L&= 2 \mathbf{v}_{ext}\mathbf{p}-2 \mathbf{v}_{ext}\tilde W^{-1}\bar {\tilde L} \bar\Gamma+\nu\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext} + \mu \mathbf{v}_{ext}\tilde W^{-1}\mathbf{w}\\
        &=2\bar\Gamma \bar{V}_{ext} -2 \mathbf{v}_{ext}\tilde W^{-1}\bar {\tilde L} \bar\Gamma+\nu\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext} + \mu \mathbf{v}_{ext}\tilde W^{-1}\mathbf{w}=0\\
        \nu &= 2\frac{\bar\Gamma \bar{V}_{ext} - \mathbf{v}_{ext}\tilde W^{-1}\bar {\tilde L} \bar\Gamma}{\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext}}+\mu\frac{\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext}}{\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext}}\\
        &= 2\frac{\bar\Gamma \bar{V}_{ext} - \mathbf{v}_{ext}\tilde W^{-1}\bar {\tilde L} \bar\Gamma}{\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext}}+\left(2\frac{\bar\Gamma\bar S - \mathbf{w}\tilde W^{-1}\bar {\tilde L} \bar\Gamma}{\mathbf{w}\tilde W^{-1}\mathbf{w}}+\nu\frac{\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext}}{\mathbf{w}\tilde W^{-1}\mathbf{w}}\right)\frac{\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext}}{\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext}}\\
        \nu\left(1-\frac{\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext}}{\mathbf{w}\tilde W^{-1}\mathbf{w}}\frac{\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext}}{\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext}}\right)&= -2\frac{\bar\Gamma \bar{V}_{ext} - \mathbf{v}_{ext}\tilde W^{-1}\bar {\tilde L} \bar\Gamma}{\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext}}+2\frac{\bar\Gamma\bar S - \mathbf{w}\tilde W^{-1}\bar {\tilde L} \bar\Gamma}{\mathbf{w}\tilde W^{-1}\mathbf{w}}\frac{\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext}}{\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext}}\\
        \nu\left(\mathbf{w}\tilde W^{-1}\mathbf{w}\cdot\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext}-(\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext})^2\right)&= -2\mathbf{w}\tilde W^{-1}\mathbf{w}\cdot\left(\bar\Gamma \bar{V}_{ext} - \mathbf{v}_{ext}\tilde W^{-1}\bar {\tilde L} \bar\Gamma\right)+2\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext}\left(\bar\Gamma\bar S - \mathbf{w}\tilde W^{-1}\bar {\tilde L} \bar\Gamma\right)\\
        \nu&= 2\frac{-\mathbf{w}\tilde W^{-1}\mathbf{w}\cdot\left(\bar{V}_{ext} - \mathbf{v}_{ext}\tilde W^{-1}\bar {\tilde L}\right)+\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext}\left(\bar S - \mathbf{w}\tilde W^{-1}\bar {\tilde L}\right)}{\mathbf{w}\tilde W^{-1}\mathbf{w}\cdot\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext}-(\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext})^2}\bar\Gamma\\
        \mu&= 2\frac{-\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext}\cdot\left(\bar S - \mathbf{w}\tilde W^{-1}\bar {\tilde L} \right)+\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext}\left( \bar{V}_{ext}- \mathbf{v}_{ext}\tilde W^{-1}\bar {\tilde L} \right)}{\mathbf{w}\tilde W^{-1}\mathbf{w}\cdot\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext}-(\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext})^2}\bar\Gamma\\
        \partial_{\mathbf p}\mathcal L&= 2\tilde{W} \mathbf{p}- 2 \bar {\tilde L} \bar\Gamma + \nu\mathbf{v}_{ext}+\mu \mathbf{w}= 0\\
        \Leftrightarrow\mathbf{p} &= \tilde W^{-1}\bar {\tilde L} \bar\Gamma - \frac{1}{2}\nu\tilde W^{-1}\mathbf{v}_{ext}-\frac{1}{2}\mu \tilde W^{-1}\mathbf{w}\\
        &= \tilde W^{-1}\left(\bar {\tilde L} - \mathbf{v}_{ext}\frac{\mathbf{w}\tilde W^{-1}\mathbf{w}\cdot\left(\bar{V}_{ext} - \mathbf{v}_{ext}\tilde W^{-1}\bar {\tilde L}\right)+\left(\bar S - \mathbf{w}\tilde W^{-1}\bar {\tilde L}\right)}{\mathbf{w}\tilde W^{-1}\mathbf{w}\cdot\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext}+(\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext})^2}-\mathbf{w}\frac{\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext}\cdot\left(\bar S - \mathbf{w}\tilde W^{-1}\bar {\tilde L} \right)+\left( \bar{V}_{ext}- \mathbf{v}_{ext}\tilde W^{-1}\bar {\tilde L} \right)}{\mathbf{w}\tilde W^{-1}\mathbf{w}\cdot\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext}+(\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext})^2}\right)\bar\Gamma\\
        &= \bar{\mathbf{P}}\bar\Gamma\\
        \end{aligned}

    Where :math:`\bar{\mathbf{P}}` is a three index tensor only dependent on the geometry of the molecule.

    Args:
        W_coulomb: 2-center overlap matrix
        L_coulomb: 3-center overlap matrix
        basis_integrals: integrals over the basis functions
        v_external_p: the density coefficients of the external potential
        v_external_C: the orbital coefficients of the external potential
        overlap_matrix: 2-center overlap matrix
        gamma: coefficients of the density in the orbital basis

    Returns:
        :math:`\bar{\mathbf{P}}`: the density coefficients in the new space
    """
    L_gamma = np.einsum("ijk,jk->i", L_coulomb, gamma, optimize=True)
    V_ext_gamma = np.einsum("ij,ij->", v_external_C, gamma, optimize=True)
    overlap_gamma = n_electrons  # np.einsum("ij,ij->", overlap_matrix, gamma, optimize=True)
    W_inv_w = np.linalg.lstsq(W_coulomb, basis_integrals, rcond=None)[0]
    w_W_inv_w = basis_integrals @ W_inv_w
    w_W_inv_L = W_inv_w @ L_gamma
    w_W_inv_v = W_inv_w @ v_external_p
    W_inv_v = np.linalg.lstsq(W_coulomb, v_external_p, rcond=None)[0]
    v_W_inv_L = W_inv_v @ L_gamma
    v_W_inv_v = W_inv_v @ v_external_p
    denuminator = w_W_inv_w * v_W_inv_v - w_W_inv_v**2
    mat_nu = (
        -w_W_inv_w * (V_ext_gamma - v_W_inv_L) + w_W_inv_v * (overlap_gamma - w_W_inv_L)
    ) / denuminator
    mat_mu = (
        -v_W_inv_v * (overlap_gamma - w_W_inv_L) + w_W_inv_v * (V_ext_gamma - v_W_inv_L)
    ) / denuminator
    right_side_eq = L_gamma - v_external_p * mat_nu - basis_integrals * mat_mu
    coeffs = np.linalg.lstsq(W_coulomb, right_side_eq, rcond=None)[0]
    return coeffs


def get_density_fitting_map(
    method_name: str,
    mol_orbital_basis: gto.Mole,
    mol_density_basis: gto.Mole,
    W_coulomb: np.ndarray,
    L_coulomb: np.ndarray,
    v_external_C: np.ndarray,
    v_external_p: np.ndarray,
) -> np.ndarray:
    r"""Constructs the (n_auxmol,n_mol,n_mol) tensor which describes a linear map from ks-to of-
    coefficients.

    .. math::

        p^i = \mathbf{P}^i_{jk}\Gamma^{jk}

    The tensor is based on the method to choose the coefficients such that they optimize a target.
    The advantage on calculation the linear map from old to new coefficients is that the leased square functions has
    only to be evaluated once for all iterations, but calculating the map takes way longer than calculating a
    single coefficient. Only worth it if there are very many coefficients to transform for a single molecule.
    """
    if method_name == "hartree+external":
        density_fitting_map = get_density_fitting_map_hartree_external(
            W_coulomb, L_coulomb, v_external_p, v_external_C
        )
    elif method_name == "hartree":
        density_fitting_map = get_density_fitting_map_hartree(W_coulomb, L_coulomb)
    elif method_name == "hartree+external_mofdft":
        density_fitting_map = get_density_fitting_map_mofdft(
            W_coulomb, L_coulomb, v_external_p, v_external_C
        )
    elif method_name == "hartree+external_mofdft_fixed_density":
        basis_integrals = get_normalization_vector(mol_density_basis)
        S_overlap = get_overlap_matrix(mol_orbital_basis)
        density_fitting_map = get_density_fitting_map_mofdft_fixed_density(
            W_coulomb, L_coulomb, v_external_p, v_external_C, basis_integrals, S_overlap
        )
    elif method_name == "overlap":
        W_overlap = get_overlap_matrix(mol_density_basis)
        L_overlap = get_overlap_tensor(mol_density_basis, mol_orbital_basis)
        density_fitting_map = get_density_fitting_map_hartree(W_overlap, L_overlap)
    elif method_name == "hartree+external_fixed_density":
        basis_integrals = get_normalization_vector(mol_density_basis)
        S_overlap = get_overlap_matrix(mol_orbital_basis)
        density_fitting_map = get_density_fitting_map_hartree_external_fixed_density(
            W_coulomb,
            L_coulomb,
            v_external_p,
            v_external_C,
            basis_integrals,
            S_overlap,
        )
    elif method_name == "hartree_fixed_density_external":
        basis_integrals = get_normalization_vector(mol_density_basis)
        S_overlap = get_overlap_matrix(mol_orbital_basis)
        density_fitting_map = get_density_fitting_map_hartree_fixed_density_external(
            W_coulomb, L_coulomb, v_external_p, v_external_C, basis_integrals, S_overlap
        )
    elif method_name == "overlap+external_fixed_density":
        W_overlap = get_overlap_matrix(mol_density_basis)
        L_overlap = get_overlap_tensor(mol_density_basis, mol_orbital_basis)
        basis_integrals = get_normalization_vector(mol_density_basis)
        S_overlap = get_overlap_matrix(mol_orbital_basis)
        density_fitting_map = get_density_fitting_map_hartree_external_fixed_density(
            W_overlap,
            L_overlap,
            v_external_p,
            v_external_C,
            basis_integrals,
            S_overlap,
        )
    elif method_name == "overlap_fixed_density_external":
        W_overlap = get_overlap_matrix(mol_density_basis)
        L_overlap = get_overlap_tensor(mol_density_basis, mol_orbital_basis)
        basis_integrals = get_normalization_vector(mol_density_basis)
        S_overlap = get_overlap_matrix(mol_orbital_basis)
        density_fitting_map = get_density_fitting_map_hartree_fixed_density_external(
            W_overlap, L_overlap, v_external_p, v_external_C, basis_integrals, S_overlap
        )
    else:
        raise NotImplementedError(f"Unknown density fitting method: {method_name}")
    return density_fitting_map


def get_density_fitting_map_hartree(W_coulomb: np.ndarray, L_coulomb: np.ndarray) -> np.ndarray:
    r""" Constructs the (n_auxmol,n_mol,n_mol) tensor which describes a linear map from ks-to of-coefficients.
    The target to optimize is the Hartree energy of the residual density:

    .. math::

        \begin{aligned}
        \mathcal{L}(\mathbf{p}) &= \mathbf{p} \tilde{W} \mathbf{p} - 2 \mathbf{p}\bar {\tilde L} \bar\Gamma + \bar\Gamma \tilde{\mathbf{D}}\bar\Gamma
        \end{aligned}

    This is solved by assuming that :math:`\tilde{W}` is invertible:

    .. math::

        \begin{aligned}
        \partial_{\mathbf p}\mathcal L&= 2\tilde{W} \mathbf{p}- 2 \bar {\tilde L} \bar\Gamma=0\\
        \mathbf{p}&=\tilde{W}^{-1}\bar {\tilde L} \bar\Gamma\\
        &=\bar{\mathbf{P}} \bar\Gamma
        \end{aligned}

    Args:
        L_overlap: 3-center coulomb matrix
        W_overlap: 2-center coulomb matrix

    Returns:
        P: the 3-index tensor which maps Gamma to p
    Notes:
        Corresponds to ´´df_coeff´´¸in the MOFDFT implementation. But is probably not used by them.
        This function is also used to compute the map if one wants to minimize the L2 - norm of the residual density.
        For this W_coulomb and L_coulomb have to be replaced with W_overlap and L_overlap.
    """
    naux = L_coulomb.shape[0]
    nao = L_coulomb.shape[1]
    a = W_coulomb
    b = L_coulomb.reshape(naux, nao * nao)
    coeff = np.linalg.lstsq(a, b, rcond=None)[0]
    return coeff.reshape(naux, nao, nao)


def get_density_fitting_map_hartree_external(
    W_coulomb: np.ndarray,
    L_coulomb: np.ndarray,
    v_external_p: np.ndarray,
    v_external_C: np.ndarray,
) -> np.ndarray:
    r""" Constructs the (n_auxmol,n_mol,n_mol) tensor which describes a linear map from ks-to of-coefficients.
    The target to optimize is the sum of Hartree energy and external energy of the residual density:

    .. math::

        \begin{aligned}
        \mathcal{L}(\mathbf{p}) &= \mathbf{p} \tilde{W} \mathbf{p} - 2 \mathbf{p}\bar {\tilde L} \bar\Gamma + \bar\Gamma \tilde{\mathbf{D}}\bar\Gamma + (\mathbf{p}\mathbf{v}_{ext}-\bar\Gamma \bar{V}_{ext})^2
        \end{aligned}

    This is solved by assuming that :math:`A=\tilde{W}+\mathbf{v}_{ext}\mathbf{v}_{ext}^T` is invertible:

    .. math::

        \begin{aligned}
        \partial_{\mathbf p}\mathcal L&= 2\tilde{W} \mathbf{p}- 2 \bar {\tilde L} \bar\Gamma + 2\mathbf{v}_{ext}\mathbf{v}_{ext}^T\mathbf{p} - 2\mathbf{v}_{ext}\bar\Gamma \bar{V}_{ext}\\
        &= 2 A \mathbf{p}- 2 \bar {\tilde L} \bar\Gamma - 2\mathbf{v}_{ext}\bar\Gamma \bar{V}_{ext}=0\\
        \mathbf{p}&=A^{-1}\left(\bar {\tilde L}+\mathbf{v}_{ext}\bar{V}_{ext}\right) \bar\Gamma\\
        &=\bar{\mathbf{P}} \bar\Gamma
        \end{aligned}

    Args:
        L_overlap: 3-center coulomb matrix of the overlap of ks- and of-coefficients
        W_overlap: 2-center coulomb matrix of the of-coefficients
        v_external_p: 1 center external potential vector of the of-coefficients
        v_external_C: 1 center external potential matrix of the ks-coefficients

    Returns:
        P: the 3-index tensor which maps Gamma to p
    Notes:
        The energy lagragian is mentioned in the [M-OFDFT]_ paper but the equation they derive does not minimize it.
        This function depicts the correct solution to the minimization problem.
    """
    naux = L_coulomb.shape[0]
    nao = L_coulomb.shape[1]
    A = W_coulomb + v_external_p[:, None] @ v_external_p[None, :]
    L_effective = L_coulomb + v_external_p[:, None, None] * v_external_C[None, :, :]
    coeff = np.linalg.lstsq(A, L_effective.reshape(naux, nao * nao), rcond=None)[0]
    return coeff.reshape(naux, nao, nao)


def get_density_fitting_map_mofdft(
    W_coulomb: np.ndarray,
    L_coulomb: np.ndarray,
    v_external_p: np.ndarray,
    v_external_C: np.ndarray,
) -> np.ndarray:
    r"""Constructs the (n_auxmol,n_mol,n_mol) tensor which describes a linear map from ks-to of-coefficients.
    The target to optimize is mentioned as the sum of the Hartree energy of the residual density and the external energy:

    .. math::

        \begin{aligned}
        \mathcal{L}(\mathbf{p}) &= \mathbf{p} \tilde{W} \mathbf{p} - 2 \mathbf{p}\bar {\tilde L} \bar\Gamma + \bar\Gamma \tilde{\mathbf{D}}\bar\Gamma + (\mathbf{p}\mathbf{v}_{ext}-\bar\Gamma \bar{V}_{ext})^2
        \end{aligned}

    But the following equation does not necessarily minimize this Energy function.

    .. math::

        \left(\begin{array}{c}\tilde{W}\\v_{ext}^T\end{array}\right) \mathbf{p} =  \left(\begin{array}{c}\tilde{L} \bar{\Gamma} \\ \bar{\Gamma}\bar{V}_{ext}\end{array}\right)

    It is solved using least squares methods.

    Args:
        L_overlap: 3-center coulomb matrix of the overlap of ks- and of-coefficients
        W_overlap: 2-center coulomb matrix of the of-coefficients
        v_external_p: 1 center external potential vector of the of-coefficients
        v_external_C: 1 center external potential matrix of the ks-coefficients

    Returns:
        P: the 3-index tensor which maps Gamma to p
    Notes:
        The energy Lagragian is mentioned in the [M-OFDFT]_ paper but the equation derived does not minimize it.
        Called ``df_coeff_jext`` in the MOFDFT ìmplementation. This is the method mentioned [M-OFDFT]_ mention
        to use in their implementation.
    """
    nao, naux = (
        L_coulomb.shape[1],
        W_coulomb.shape[0],
    )
    int_1c1e_nuc = v_external_p
    a = np.concatenate([W_coulomb, int_1c1e_nuc[None]], axis=0)
    b = L_coulomb.reshape(naux, nao * nao)
    b = np.concatenate([b, v_external_C.reshape(1, -1)], axis=0)
    coeff = np.linalg.lstsq(a, b, rcond=None)[0]
    return coeff.reshape(naux, nao, nao)


def get_density_fitting_map_mofdft_fixed_density(
    W_coulomb: np.ndarray,
    L_coulomb: np.ndarray,
    v_external_p: np.ndarray,
    v_external_C: np.ndarray,
    basis_integrals: np.ndarray,
    overlap_matrix: np.ndarray,
) -> np.ndarray:
    r"""Constructs the (n_auxmol,n_mol,n_mol) tensor which describes a linear map from ks-to of-coefficients.
    The target to optimize is mentioned as the sum of the Hartree energy of the residual density and the external energy as well as the differences in L1 norm of the density:

    .. math::

        \begin{aligned}
        \mathcal{L}(\mathbf{p}) &= \mathbf{p} \tilde{W} \mathbf{p} - 2 \mathbf{p}\bar {\tilde L} \bar\Gamma + \bar\Gamma \tilde{\mathbf{D}}\bar\Gamma + (\mathbf{p}\mathbf{v}_{ext}-\bar\Gamma \bar{V}_{ext})^2 + (\mathbf{p}\mathbf{w}-\bar\Gamma \bar{S})^2
        \end{aligned}

    But the following equation does not necessarily minimizes this Energy function.

    .. math::

        \left(\begin{array}{c}\tilde{W}\\\mathbf{v}_{ext}^T\\\mathbf{w}^T\end{array}\right) \mathbf{p} =  \left(\begin{array}{c}\tilde{L} \bar{\Gamma} \\ \bar{\Gamma}\bar{V}_{ext}\\\bar S\bar \Gamma\end{array}\right)

    It is solved using least squares methods.
    Args:
        L_overlap: 3-center coulomb matrix of the overlap of ks- and of-coefficients
        W_overlap: 2-center coulomb matrix of the of-coefficients
        v_external_p: 1 center external potential vector of the of-coefficients
        v_external_C: 1 center external potential matrix of the ks-coefficients
        basis_integrals: 1 center integrals over the basis functions of the of basis
        overlap_matrix: 1 center intergrals over products of the basis functions of the ks-basis

    Returns:
        P: the 3-index tensor which maps Gamma to p
    Notes:
        The energy Lagragian is not mentioned in [M-OFDFT]_ , but it is implemented in the MOFDFT Github project.
        Called ``get_rho_coeff_jextnelec_fit`` in the MOFDFT ìmplementation.
    """
    nao, naux = (
        L_coulomb.shape[1],
        W_coulomb.shape[0],
    )
    int_1c1e_nuc = v_external_p
    a = np.concatenate([W_coulomb, int_1c1e_nuc[None], basis_integrals[None]], axis=0)
    b = L_coulomb.reshape(naux, nao * nao)
    b = np.concatenate([b, v_external_C.reshape(1, -1), overlap_matrix.reshape(1, -1)], axis=0)
    coeff = np.linalg.lstsq(a, b, rcond=None)[0]
    return coeff.reshape(naux, nao, nao)


def get_density_fitting_map_hartree_external_fixed_density(
    W_coulomb: np.ndarray,
    L_coulomb: np.ndarray,
    v_external_p: np.ndarray,
    v_external_C: np.ndarray,
    basis_integrals: np.ndarray,
    overlap_matrix: np.ndarray,
) -> np.ndarray:
    r""" Constructs the (n_auxmol,n_mol,n_mol) tensor which describes a linear map from ks-to of-coefficients.
     The target to optimize is the sum of Hartree energy and external energy of the residual density,
      the L1 norm of the density and the external energy are enforced to stay constant after the mapping:


    .. math::

        \begin{aligned}
        \mathcal{L}(\mathbf{p},\mu) &= \mathbf{p} \tilde{W} \mathbf{p} - 2 \mathbf{p}\bar {\tilde L} \bar\Gamma + (\mathbf{p}\mathbf{v}_{ext}-\bar\Gamma \bar{V}_{ext})^2+\mu(\mathbf{p}\mathbf{w}-\bar\Gamma\bar S)\\
        \partial_{\mathbf p}\mathcal L&= 2\tilde{W} \mathbf{p}- 2 \bar {\tilde L} \bar\Gamma + 2\mathbf{v}_{ext}(\mathbf{p}\mathbf{v}_{ext}-\bar\Gamma \bar{V}_{ext})+\mu \mathbf{w}\\
        &= 2(\tilde{W}+\mathbf{v}_{ext}\mathbf{v}_{ext}^T) \mathbf{p}- 2 \bar {\tilde L} \bar\Gamma + 2\mathbf{v}_{ext}\bar\Gamma \bar{V}_{ext}+\mu \mathbf{w}= 0\\
        \partial_\mu\mathcal{L}&= (\mathbf{p}\mathbf{w}-\bar\Gamma\bar S)=0\\
        \end{aligned}

    If we assume that :math:`A=\tilde{W}+\mathbf{v}_{ext}\mathbf{v}_{ext}^T` is invertible

    .. math::

        \begin{aligned}
        \mathbf{w}A^{-1}\partial_{\mathbf p}\mathcal L&= 2 \mathbf{w}\mathbf{p}-2 \mathbf{w}A^{-1}\bar {\tilde L} \bar\Gamma-2\mathbf{w}A^{-1}\mathbf{v}_{ext}\bar\Gamma \bar{V}_{ext} + \mu \mathbf{w}A^{-1}\mathbf{w}\\
        &=2\bar\Gamma\bar S -2 \mathbf{w}A^{-1}\bar {\tilde L} \bar\Gamma-2\mathbf{w}A^{-1}\mathbf{v}_{ext}\bar\Gamma \bar{V}_{ext} + \mu \mathbf{w}A^{-1}\mathbf{w}=0\\
        \mu &= 2\frac{-\bar\Gamma\bar S + \mathbf{w}A^{-1}\bar {\tilde L} \bar\Gamma+\mathbf{w}A^{-1}\mathbf{v}_{ext}\bar\Gamma \bar{V}_{ext}}{\mathbf{w}A^{-1}\mathbf{w}}\\
        &=2\frac{\mathbf{w}A^{-1}\bar {\tilde L} +\mathbf{w}A^{-1}\mathbf{v}_{ext} \bar{V}_{ext}-\bar S}{\mathbf{w}A^{-1}\mathbf{w}}\bar\Gamma\\
        \partial_{\mathbf p}\mathcal L&= 2A\mathbf{p}- 2 \bar {\tilde L} \bar\Gamma - 2\mathbf{v}_{ext}\bar\Gamma \bar{V}_{ext}+\mu \mathbf{w}=0\\
        \Leftrightarrow\mathbf p&=A^{-1}\bar {\tilde L} \bar\Gamma + A^{-1}\mathbf{v}_{ext}\bar\Gamma \bar{V}_{ext}-\frac{1}{2}\mu A^{-1}\mathbf{w}\\
        &=A^{-1}\bar {\tilde L} \bar\Gamma + A^{-1}\mathbf{v}_{ext}\bar\Gamma \bar{V}_{ext}- A^{-1}\mathbf{w}\frac{\mathbf{w}A^{-1}\bar {\tilde L} +\mathbf{w}A^{-1}\mathbf{v}_{ext} \bar{V}_{ext}-\bar S}{\mathbf{w}A^{-1}\mathbf{w}}\bar\Gamma\\
        &=A^{-1}\left(\bar {\tilde L} + \mathbf{v}_{ext} \bar{V}_{ext}- \mathbf{w}\frac{\mathbf{w}A^{-1}\bar {\tilde L} +\mathbf{w}A^{-1}\mathbf{v}_{ext} \bar{V}_{ext}-\bar S}{\mathbf{w}A^{-1}\mathbf{w}}\right)\bar\Gamma\\
        &= \bar {\mathbf{P}}\bar\Gamma
        \end{aligned}

    Where :math:`\bar{\mathbf{P}}` is a three index tensor only dependent on the geometry of the molecule.

    Args:
        W_coulomb: 2-center overlap matrix
        L_coulomb: 3-center overlap matrix
        basis_integrals: integrals over the basis functions
        v_external_p: the density coefficients of the external potential
        v_external_C: the orbital coefficients of the external potential
        overlap_matrix: 2-center overlap matrix


    Returns:
        :math:`\bar{\mathbf{P}}`: the density coefficients in the new space
    """
    naux, nao = L_coulomb.shape[0], L_coulomb.shape[1]
    A = W_coulomb + v_external_p[:, None] @ v_external_p[None, :]
    A_inv_w = np.linalg.lstsq(A, basis_integrals, rcond=None)[0]
    w_A_inv_w = basis_integrals @ A_inv_w
    w_A_inv_L = np.einsum("i,ijk->jk", A_inv_w, L_coulomb, optimize=True)
    w_A_inv_v = A_inv_w @ v_external_p
    matrix = (w_A_inv_L + w_A_inv_v * v_external_C - overlap_matrix) / w_A_inv_w
    right_side_eq = (
        L_coulomb
        + v_external_p[:, None, None] * v_external_C[None, :, :]
        - basis_integrals[:, None, None] * matrix[None, :, :]
    )
    P = np.linalg.lstsq(A, right_side_eq.reshape(naux, nao * nao), rcond=None)[0].reshape(
        naux, nao, nao
    )
    return P


def get_density_fitting_map_hartree_fixed_density_external(
    W_coulomb: np.ndarray,
    L_coulomb: np.ndarray,
    v_external_p: np.ndarray,
    v_external_C: np.ndarray,
    basis_integrals: np.ndarray,
    overlap_matrix: np.ndarray,
) -> np.ndarray:
    r""" Constructs the (n_auxmol,n_mol,n_mol) tensor which describes a linear map from ks-to of-coefficients.
    The target to optimize is the Hartree energy of the residual density,
    the L1 norm of the density and the external energy are enforced to stay constant after the mapping:

    .. math::

        \begin{aligned}
        \mathcal{L}(\mathbf{p},\mu,\nu) &= \mathbf{p} \tilde{W} \mathbf{p} - 2 \mathbf{p}\bar {\tilde L} \bar\Gamma + \nu(\mathbf{p}\mathbf{v}_{ext}-\bar\Gamma \bar{V}_{ext})+\mu(\mathbf{p}\mathbf{w}-\bar\Gamma\bar S)\\
        \partial_{\mathbf p}\mathcal L&= 2\tilde{W} \mathbf{p}- 2 \bar {\tilde L} \bar\Gamma + \nu\mathbf{v}_{ext}+\mu \mathbf{w}= 0\\
        \partial_\mu\mathcal{L}&= (\mathbf{p}\mathbf{w}-\bar\Gamma\bar S)=0\\
        \partial_\nu\mathcal{L}&= (\mathbf{p}\mathbf{v}_{ext}-\bar\Gamma \bar{V}_{ext})=0\\
        \end{aligned}

    If we assume that :math:`A=\tilde{W}` is invertible

    .. math::
        \begin{aligned}
        \mathbf{w}\tilde W^{-1}\partial_{\mathbf p}\mathcal L&= 2 \mathbf{w}\mathbf{p}-2 \mathbf{w}\tilde W^{-1}\bar {\tilde L} \bar\Gamma+\nu\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext} + \mu \mathbf{w}\tilde W^{-1}\mathbf{w}\\
        &=2\bar\Gamma\bar S -2 \mathbf{w}\tilde W^{-1}\bar {\tilde L} \bar\Gamma+\nu\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext} + \mu \mathbf{w}\tilde W^{-1}\mathbf{w}=0\\
        \mu &= 2\frac{\bar\Gamma\bar S - \mathbf{w}\tilde W^{-1}\bar {\tilde L} \bar\Gamma}{\mathbf{w}\tilde W^{-1}\mathbf{w}}+\nu\frac{\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext}}{\mathbf{w}\tilde W^{-1}\mathbf{w}}\\
        \mathbf{v}_{ext}\tilde W^{-1}\partial_{\mathbf p}\mathcal L&= 2 \mathbf{v}_{ext}\mathbf{p}-2 \mathbf{v}_{ext}\tilde W^{-1}\bar {\tilde L} \bar\Gamma+\nu\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext} + \mu \mathbf{v}_{ext}\tilde W^{-1}\mathbf{w}\\
        &=2\bar\Gamma \bar{V}_{ext} -2 \mathbf{v}_{ext}\tilde W^{-1}\bar {\tilde L} \bar\Gamma+\nu\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext} + \mu \mathbf{v}_{ext}\tilde W^{-1}\mathbf{w}=0\\
        \nu &= 2\frac{\bar\Gamma \bar{V}_{ext} - \mathbf{v}_{ext}\tilde W^{-1}\bar {\tilde L} \bar\Gamma}{\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext}}+\mu\frac{\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext}}{\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext}}\\
        &= 2\frac{\bar\Gamma \bar{V}_{ext} - \mathbf{v}_{ext}\tilde W^{-1}\bar {\tilde L} \bar\Gamma}{\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext}}+\left(2\frac{\bar\Gamma\bar S - \mathbf{w}\tilde W^{-1}\bar {\tilde L} \bar\Gamma}{\mathbf{w}\tilde W^{-1}\mathbf{w}}+\nu\frac{\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext}}{\mathbf{w}\tilde W^{-1}\mathbf{w}}\right)\frac{\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext}}{\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext}}\\
        \nu\left(1-\frac{\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext}}{\mathbf{w}\tilde W^{-1}\mathbf{w}}\frac{\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext}}{\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext}}\right)&= -2\frac{\bar\Gamma \bar{V}_{ext} - \mathbf{v}_{ext}\tilde W^{-1}\bar {\tilde L} \bar\Gamma}{\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext}}+2\frac{\bar\Gamma\bar S - \mathbf{w}\tilde W^{-1}\bar {\tilde L} \bar\Gamma}{\mathbf{w}\tilde W^{-1}\mathbf{w}}\frac{\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext}}{\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext}}\\
        \nu\left(\mathbf{w}\tilde W^{-1}\mathbf{w}\cdot\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext}-(\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext})^2\right)&= -2\mathbf{w}\tilde W^{-1}\mathbf{w}\cdot\left(\bar\Gamma \bar{V}_{ext} - \mathbf{v}_{ext}\tilde W^{-1}\bar {\tilde L} \bar\Gamma\right)+2\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext}\left(\bar\Gamma\bar S - \mathbf{w}\tilde W^{-1}\bar {\tilde L} \bar\Gamma\right)\\
        \nu&= 2\frac{-\mathbf{w}\tilde W^{-1}\mathbf{w}\cdot\left(\bar{V}_{ext} - \mathbf{v}_{ext}\tilde W^{-1}\bar {\tilde L}\right)+\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext}\left(\bar S - \mathbf{w}\tilde W^{-1}\bar {\tilde L}\right)}{\mathbf{w}\tilde W^{-1}\mathbf{w}\cdot\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext}-(\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext})^2}\bar\Gamma\\
        \mu&= 2\frac{-\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext}\cdot\left(\bar S - \mathbf{w}\tilde W^{-1}\bar {\tilde L} \right)+\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext}\left( \bar{V}_{ext}- \mathbf{v}_{ext}\tilde W^{-1}\bar {\tilde L} \right)}{\mathbf{w}\tilde W^{-1}\mathbf{w}\cdot\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext}-(\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext})^2}\bar\Gamma\\
        \partial_{\mathbf p}\mathcal L&= 2\tilde{W} \mathbf{p}- 2 \bar {\tilde L} \bar\Gamma + \nu\mathbf{v}_{ext}+\mu \mathbf{w}= 0\\
        \Leftrightarrow\mathbf{p} &= \tilde W^{-1}\bar {\tilde L} \bar\Gamma - \frac{1}{2}\nu\tilde W^{-1}\mathbf{v}_{ext}-\frac{1}{2}\mu \tilde W^{-1}\mathbf{w}\\
        &= \tilde W^{-1}\left(\bar {\tilde L} - \mathbf{v}_{ext}\frac{\mathbf{w}\tilde W^{-1}\mathbf{w}\cdot\left(\bar{V}_{ext} - \mathbf{v}_{ext}\tilde W^{-1}\bar {\tilde L}\right)+\left(\bar S - \mathbf{w}\tilde W^{-1}\bar {\tilde L}\right)}{\mathbf{w}\tilde W^{-1}\mathbf{w}\cdot\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext}+(\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext})^2}-\mathbf{w}\frac{\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext}\cdot\left(\bar S - \mathbf{w}\tilde W^{-1}\bar {\tilde L} \right)+\left( \bar{V}_{ext}- \mathbf{v}_{ext}\tilde W^{-1}\bar {\tilde L} \right)}{\mathbf{w}\tilde W^{-1}\mathbf{w}\cdot\mathbf{v}_{ext}\tilde W^{-1}\mathbf{v}_{ext}+(\mathbf{w}\tilde W^{-1}\mathbf{v}_{ext})^2}\right)\bar\Gamma\\
        &= \bar{\mathbf{P}}\bar\Gamma\\
        \end{aligned}

    Where :math:`\bar{\mathbf{P}}` is a three index tensor only dependent on the geometry of the molecule.

    Args:
        W_coulomb: 2-center overlap matrix
        L_coulomb: 3-center overlap matrix
        basis_integrals: integrals over the basis functions
        v_external_p: the density coefficients of the external potential
        v_external_C: the orbital coefficients of the external potential
        overlap_matrix: 2-center overlap matrix


    Returns:
        :math:`\bar{\mathbf{P}}`: the density coefficients in the new space
    """
    naux, nao = L_coulomb.shape[0], L_coulomb.shape[1]
    W_inv_w = np.linalg.lstsq(W_coulomb, basis_integrals, rcond=None)[0]
    w_W_inv_w = basis_integrals @ W_inv_w
    w_W_inv_L = np.einsum("i,ijk->jk", W_inv_w, L_coulomb, optimize=True)
    w_W_inv_v = W_inv_w @ v_external_p
    W_inv_v = np.linalg.lstsq(W_coulomb, v_external_p, rcond=None)[0]
    v_W_inv_L = np.einsum("i,ijk->jk", W_inv_v, L_coulomb, optimize=True)
    v_W_inv_v = W_inv_v @ v_external_p
    denuminator = w_W_inv_w * v_W_inv_v - w_W_inv_v**2
    mat_nu = (
        -w_W_inv_w * (v_external_C - v_W_inv_L) + w_W_inv_v * (overlap_matrix - w_W_inv_L)
    ) / denuminator
    mat_mu = (
        -v_W_inv_v * (overlap_matrix - w_W_inv_L) + w_W_inv_v * (v_external_C - v_W_inv_L)
    ) / denuminator
    right_side_eq = (
        L_coulomb
        - v_external_p[:, None, None] * mat_nu[None, :, :]
        - basis_integrals[:, None, None] * mat_mu[None, :, :]
    )
    P = np.linalg.lstsq(W_coulomb, right_side_eq.reshape(naux, nao * nao), rcond=None)[0].reshape(
        naux, nao, nao
    )
    return P


def density_fitting_mol(
    gamma: np.ndarray,
    mol_orbital: gto.Mole,
    mol_density: gto.Mole,
    method="hartree+external_mofdft",
) -> np.ndarray:
    """Calculates all needed basis integrals and returns density coefficients.

    This is a wrapper around :py:func:`density_fitting` which calculates the necessary
    integrals given the molecule objects and the density matrix in the orbital basis.

    Args:
        gamma: the density matrix in the orbital basis
        mol_orbital: the molecule object containing information about the orbital basis
        mol_rho: the molecule object containing information about the density basis

    Returns:
        coeffs: the density coefficients in the density basis

    Raises:
        AssertionError: if the two molecule geometries are not the same
        AssertionError: if the number of electrons is not conserved
    """
    assert gto.mole.is_same_mol(
        mol_density, mol_orbital, cmp_basis=False
    ), "Density fitting was given different molecules/geometries in mol_rho and mol_orbital"

    coulomb_matrix = basis_integrals.get_coulomb_matrix(mol_density)
    nuclear_attraction_vector = basis_integrals.get_nuclear_attraction_vector(mol_density)
    nuclear_attraction_matrix = basis_integrals.get_nuclear_attraction_matrix(mol_orbital)

    density_fit_function = get_density_fitting_function(
        method,
        mol_orbital,
        mol_density,
        W_coulomb=coulomb_matrix,
        v_external_C=nuclear_attraction_matrix,
        v_external_p=nuclear_attraction_vector,
    )
    coeffs = density_fit_function(gamma)
    return coeffs


T = TypeVar("T", np.ndarray, torch.Tensor)


def ksdft_density_matrix(molecular_orbital_coefficients: T, occupation_numbers: T) -> T:
    r"""Calculates the density matrix from the coefficients of the molecular orbitals.

    The molecular orbitals are given by
    :math:`| \phi_i \rangle = \sum_{\alpha} C_{\alpha i} | \eta_ \alpha \rangle`. The density
    matrix is then calculated as

    .. math::

        \Gamma_{ \alpha \beta} = \sum_{i=1}^{n_{MO}} n_i C_{\alpha i} C_{i \beta}^\dagger,

    where :math:`n_i` are the occupation numbers of the orbital :math:`\phi_i`. In restricted
    KS, :math:`n_i = 2` for occupied orbitals and :math:`n_i = 0` for unoccupied orbitals.

    Args:
        molecular_orbital_coefficients: the coefficients of the orbitals in the basis
        occupation_numbers: the occupation number of the orbitals

    Returns:
        gamma: the density matrix in the orbital basis
    """
    mo_occupied = molecular_orbital_coefficients[:, occupation_numbers > 0]
    gamma = mo_occupied * occupation_numbers[occupation_numbers > 0] @ mo_occupied.conj().T
    return gamma


def get_KSDFT_Hartree_potential(
    mol: gto.Mole, gamma: np.ndarray, hermitian: int = 1, **kwargs
) -> np.ndarray:
    r"""Wrapper around the get_j function from pyscf which returns the potential matrix for the
    Hartree potential. Necessary for checking the quality of the density optimisation.

    .. math::

        J_{\alpha,\beta} = (\eta_\alpha \eta_\beta,\eta_\gamma\eta_\delta) \Gamma_{\gamma,\delta}

    Args:
        mol: The Molecule object which contains information over the used basis
        gamma: the density matrix in the orbital basis
        kwargs: keyword arguments passed to pyscf.dft.RKS.get_j
        hermitian: whether the matrix is hermitian (0: no symmetry, 1: hermitian, -1: anti-hermitian)

    Returns:
        v_hart: the hartree potential in the orbital basis shape (n_b,n_b)
    """
    return dft.RKS(mol).get_j(mol, gamma, hermitian, **kwargs)
