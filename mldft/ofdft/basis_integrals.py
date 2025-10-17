"""Compute analytic integrals over contracted Gaussian-type orbitals (GTO).

This module uses pyscf which in turn uses libcint to compute analytic
integrals. Basis functions are usually separated into radial and
spherical parts,

.. math::

        \\omega_\\mu(r, \\theta, \\phi) = f_{l_\\mu}(r) Y_{l_\\mu}^{m_\\mu} (\\theta, \\phi),

where :math:`Y_l^m` are spherical harmonics. The radial part of contracted
Gaussian-type orbitals are given by a monomial times a linear combination of
Gaussians,

.. math::

        f_l(r) = r^l \\sum_{i=1}^N c_i \\exp(-\\zeta_i r^2).

The :math:`c_i` are called contraction coefficients and the :math:`\\zeta_i` scale the
width of the Gaussians. :math:`N` is the number of primitives. Note that also the
nucleus can be modeled using GTOs (:code:`nuc_model=1` is point particle).

`pyscf` molecules have three layers of basis data (see
https://pyscf.org/develop/gto_developer.html):

*   input, e.g. :code:`.atom`, :code:`.basis`
*   internal, e.g. :code:`._atom`, :code:`._base`
*   for :code:`libcint`, e.g. :code:`._atm`, :code:`._bas`, :code:`._env`

The ._env array stores all relevant information about the basis set,
including the atom positions, the exponents and contraction coefficients.
The properties :code:`._atm` and :code:`._bas` are used to store the indices of the :code:`._env`
array. They contain the following information (0 are entries not relevant for
us):

*   :code:`_atm`: charge, ind_of_coord, nuc_model, ind_nuc_zeta, 0, 0
*   :code:`_bas`: atom_id, angular_momentum, num_primitives, num_contracted_gtos, 0, ind_exp, ind_contraction_coeff, 0
"""

from functools import lru_cache
from typing import Tuple

import numpy as np
from pyscf import dft, gto

from mldft.utils.einsum import einsum

# correction factor for integrals containing the flat Gaussian
FLAT_GAUSSIAN_CORRECTION = 2 * np.sqrt(np.pi)


def _append_flat_gaussian(bas, env) -> Tuple[np.ndarray, np.ndarray]:
    """Append a flat Gaussian to the basis set.

    The added flat Gaussian (constant function with value 1) is needed to
    compute integrals using pyscf.

    Args:
        bas: The basis array as used by libcint.
        env: The environment array as used by libcint.

    Returns:
        bas_mod: The modified basis array.
        env_mod: The modified environment array.
    """
    # add exponent 0 and contraction coefficient 1
    env_mod = np.append(env, [0, 1])

    ind_exp = env_mod.size - 2
    ind_ctrc_coef = env_mod.size - 1

    bas_mod = np.zeros((bas.shape[0] + 1, bas.shape[1]), dtype=np.int32)
    bas_mod[:-1, :] = bas
    bas_mod[-1] = [0, 0, 1, 1, 0, ind_exp, ind_ctrc_coef, 0]

    return bas_mod, env_mod


def _get_one_center_integral(mol: gto.Mole, intor_name: str, components: int = 1) -> np.ndarray:
    """Compute 1-center integrals.

    Args:
        mol: The molecule containing information about the geometry and the
            basis set.
        intor_name: The name of the integral to compute. Partial list at
            `pyscf <https://pyscf.org/pyscf_api_docs/pyscf.gto.html#module-pyscf.gto.moleintor>`_.
            More options in the c++ library at
            `libcint <https://github.com/sunqm/libcint/blob/master/scripts/auto_intor.cl>`_.
        components: The number of components of the integral. For example, the
            dipole moment has three components.

    Returns:
        Vector of integrals.
    """

    atm = mol._atm
    bas = mol._bas
    env = mol._env

    bas, env = _append_flat_gaussian(bas, env)
    # add one of ('_sph', '_cart', '_spinor', '_ssc') suffixes
    intor_name = mol._add_suffix(intor_name)

    nbas = bas.shape[0]

    # shell slice: choose indices of the resulting tensor,
    # in this case of the matrix <\omega_i | [operator] | \omega_j>
    # shls_slice = (ish_start, ish_end, jsh_start, jsh_end)
    # we want the true basis functions as bra and the flat gaussian as ket
    shls_slice = (0, nbas - 1, nbas - 1, nbas)

    integral = gto.moleintor.getints2c(
        intor_name, atm, bas, env, shls_slice=shls_slice, comp=components
    )
    integral = np.squeeze(integral, axis=-1)

    return integral * FLAT_GAUSSIAN_CORRECTION


def get_normalization_vector(mol: gto.Mole, clip_minimum: float = 1.6e-15) -> np.ndarray:
    r"""Compute the integral over the basis functions.

    Compute

    .. math::

        w_\mu = \int \omega_\mu(r) \mathrm{d}r

    for all basis functions :math:`\omega_\mu`.

    Args:
        mol: The molecule containing information about the geometry and the
            basis set.
        clip_minimum: If not None, the integral is clipped from below at this threshold.
            This should be used to avoid numerical instabilities which can lead to larger
            errors later on.

    Returns:
        Vector of integrals.
    """
    integral = _get_one_center_integral(mol, "int1e_ovlp")
    if clip_minimum is not None:
        integral[np.abs(integral) < clip_minimum] = 0
    return integral


@lru_cache
def get_basis_dipole(mol: gto.Mole) -> np.ndarray:
    r"""Compute the basis dipole vector.

    Compute

    .. math::

        d_{i \mu} = \int \omega_\mu(r) r_i \mathrm{d}r

    for all basis functions :math:`\omega_\mu` and coordinates :math:`r_i`.

    Args:
        mol: The molecule containing information about the geometry and the
            basis set.

    Returns:
        Matrix of integrals. With shape (3, n_basis).
    """
    return _get_one_center_integral(mol, "int1e_r", components=3)


def get_potential_vector(ao: np.ndarray, potential: np.ndarray, grid: dft.Grids) -> np.ndarray:
    r"""Compute the potential vector for given potential on the grid.

    For a given potential :math:`v_\mathrm{pot}` and for all atomic orbitals
    :math:`\omega_\mu` compute

    .. math::
        v_\mu = \sum_{i \in \mathrm{grid}} (\omega_\mu)_i (v_\mathrm{pot})_i w_i
        \approx \int \omega_\mu(r) v_\mathrm{pot}(r) \mathrm{d}r

    on the grid. The weights :math:`w_i` are taken from ``grids``.

    Args:
        ao: The atomic orbitals on the grid with size (n_grid, n_ao). If the
            atomic orbitals also contain derivatives in an array of size (:,
            n_grid, n_ao), they are ignored.
        potential: The exchange-correlation potential on the grid.
        grid: The grid on which the atomic orbitals and the
            exchange-correlation potential are defined.

    Returns:
        Vector of integrals.
    """
    if ao.ndim > 2:
        ao = ao[0]  # throw away derivatives
    integral = einsum("ni,n,n->i", ao, potential, grid.weights)
    return integral


def get_gga_potential_matrix(ao: np.ndarray, vsigma: np.ndarray, grid: dft.Grids) -> np.ndarray:
    r"""Compute the potential matrix for given GGA potential on the grid.

    Computes the potential matrix :math:`V^{(\sigma)}` from the functional derivative w.r.t.
    :math:`\sigma:=\nabla\rho(r)\nabla\rho(r)`. This is necessary for the calculation of the
    correct gradient of the K/XC functionals. Their gradient can be separated into the usual
    LDA like gradient and a GGA gradient contribution.

    .. math::
        \nabla_p E_\mathrm{XC} = \nabla_p E_\mathrm{LDA} + \nabla_p E_\mathrm{GGA}


    The GGA gradient contribution is given by

    .. math::
        \nabla_p E_\mathrm{GGA} = \int\underbrace{\frac{\delta E[\rho]}{\delta\sigma(r)}}_{
        v_\sigma(r)}\nabla_p \sigma(r)\mathrm{d}r,

    which can be rewritten as

    .. math::
        (\nabla_p E_\mathrm{GGA})_\mu &= 2 \sum_\nu p_\nu \underbrace{\int v_\sigma(
        r)\nabla\omega_\mu(r)\nabla\omega_\nu(r)\mathrm{d}r}_{V^{(\sigma)}_{\mu\nu}}, \\
        \nabla_p E_\mathrm{GGA} &= 2 V^{(\sigma)} p.

    This function computes the matrix

    .. math::

        V^{(\sigma)}_{\mu\nu} = \int v_\sigma(r)\nabla\omega_\mu(r)\nabla\omega_\nu(r)\mathrm{d}r.

    Args:
        ao: The atomic orbitals on the grid with size (n_derivative, n_grid, n_ao).
        vsigma: The functional derivative w.r.t :math:`\sigma:=\nabla rho \nabla \rho` on the grid.
        grid: The grid on which the atomic orbitals and the functional are evaluated.

    Returns:
        Matrix of integrals.
    """
    assert ao.ndim > 2, "AO derivatives are required, ao dimension must be greter 2"
    assert ao.shape[0] >= 4, "x,y,z derivatives required; ao shape must be at least 4"
    nabla_ao = ao[1:4]
    mat = einsum("pni,n,pnj", nabla_ao, vsigma * grid.weights, nabla_ao)
    return mat


def get_nuclear_attraction_vector(mol: gto.Mole) -> np.ndarray:
    r"""Compute the nuclear attraction integral vector.

    Compute the integral over the potential of the nuclei :math:`a`

    .. math::

        (v_\mathrm{nuc})_\mu = -\int \omega_\mu(r) \sum_{a=1}^A \frac{Z_a}{\|r - r_a\|} \mathrm{d}r

    for all basis functions :math:`\omega_\mu`.

    Args:
        mol: The molecule containing information about the geometry and the
            basis set.

    Returns:
        Vector of integrals.
    """
    return _get_one_center_integral(mol, "int1e_nuc")


def get_nuclear_gradient_vector(mol: gto.Mole) -> np.ndarray:
    r"""Compute the nuclear gradient integral vector.

    Compute the integral over the gradient of the potential of the nuclei :math:`a`

    .. math::

        \left(\nabla_{R_A}v_\mathrm{nuc}\right)_{ A_{x/y/z}\mu} = -\int \omega_\mu(r) \nabla_{R_A} \frac{Z_A}{\|r - r_A\|} \mathrm{d}r

    for all basis functions :math:`\omega_\mu` and nuclei :math:`A`.

    Args:
        mol: The molecule containing information about the geometry and the
            basis set.

    Returns:
        Vector of integrals, shape (3 * natms, nao), ordered according to atom id.
    """
    intor_name = "int1e_iprinv"
    components = 3

    # add one of ('_sph', '_cart', '_spinor', '_ssc') suffixes
    intor_name = mol._add_suffix(intor_name)

    nuclear_gradient = np.zeros((3 * mol.natm, mol.nao), dtype=np.float64)

    for atm_id in range(mol.natm):
        with mol.with_rinv_as_nucleus(atm_id):
            atm = mol._atm
            bas = mol._bas
            env = mol._env

            bas, env = _append_flat_gaussian(bas, env)

            nbas = bas.shape[0]

            # shell slice: choose indices of the resulting tensor,
            # in this case of the matrix <\omega_i | [operator] | \omega_j>
            # shls_slice = (ish_start, ish_end, jsh_start, jsh_end)
            # we want the true basis functions as bra and the flat gaussian as ket
            shls_slice = (0, nbas - 1, nbas - 1, nbas)
            integral = gto.moleintor.getints2c(
                intor_name, atm, bas, env, shls_slice=shls_slice, comp=components
            )
            integral = np.squeeze(integral, axis=-1)
            integral *= -mol.atom_charge(atm_id)
            integral *= FLAT_GAUSSIAN_CORRECTION

            nuclear_gradient[3 * atm_id : 3 * atm_id + 3, :] = integral

    return nuclear_gradient


def get_nuclear_attraction_matrix(mol: gto.Mole) -> np.ndarray:
    r"""Compute the 2-center nuclear attraction integral matrix.

    Compute

    .. math::

        (V_\mathrm{nuc})_{\mu\nu} = \int \eta_\mu(r) \sum_{a=1}^A  \frac{Z_a}{\|r - r_a\|}
        \eta_\nu(r) \mathrm{d} r

    for all basis functions :math:`\eta_\mu` and :math:`\eta_\nu`.

    Args:
        mol: The molecule containing information about the geometry and the
            basis set.

    Returns:
        Matrix of integrals.
    """
    intor_name = mol._add_suffix("int1e_nuc")
    return gto.moleintor.getints(intor_name, mol._atm, mol._bas, mol._env, hermi=1)


def get_nuclear_gradient_matrix(mol: gto.Mole) -> np.ndarray:
    r"""Compute the 2-center nuclear gradient integral matrix.

    Compute

    .. math::

        \left(\nabla_{R_A}V_\mathrm{nuc}\right)_{A_{x/y/z}\mu\nu} = \int \omega_\mu(r) \left(\nabla_{R_A}\frac{Z_A}{\|r_1 - A_A\|} \right) \omega_\nu(r)
        \mathrm{d} r

    for all basis functions :math:`\omega_\mu` and :math:`\omega_\nu` and all nuclei :math:`A`.

    Args:
        mol: The molecule containing information about the geometry and the
            basis set.

    Returns:
        Matrix of integrals shape (3 * natoms, nao, nao), ordered according to atom id.
    """
    intor_name = "int1e_iprinv"
    components = 3

    # add one of ('_sph', '_cart', '_spinor', '_ssc') suffixes
    intor_name = mol._add_suffix(intor_name)

    nuclear_gradient = np.zeros((3 * mol.natm, mol.nao, mol.nao), dtype=np.float64)

    for atm_id in range(mol.natm):
        with mol.with_rinv_as_nucleus(atm_id):
            integral = mol.intor(intor_name, comp=components)
            integral *= -mol.atom_charge(atm_id)
            nuclear_gradient[3 * atm_id : 3 * atm_id + 3, :, :] = integral + np.swapaxes(
                integral, 1, 2
            )

    return nuclear_gradient


def get_overlap_matrix(mol: gto.Mole) -> np.ndarray:
    r"""Compute the 2-center overlap matrix.

    Compute

    .. math::

        W_{\mu\nu} = \int \omega_\mu(r) \omega_\nu(r) \mathrm{d}r

    for all basis functions :math:`\omega_\mu` and :math:`\omega_\nu`.

    Args:
        mol: The molecule containing information about the geometry and the
            basis set.

    Returns:
        Matrix of integrals.
    """
    intor_name = mol._add_suffix("int1e_ovlp")
    return gto.moleintor.getints(intor_name, mol._atm, mol._bas, mol._env)


def get_overlap_tensor(mol_rho: gto.Mole, mol_orbital: gto.Mole) -> np.ndarray:
    r"""Compute the 3-center overlap integral tensor between different bases.

    Compute the overlap integral over one atomic orbital :math:`\omega_\mu` of the density basis
    given by :code:`mol_rho` and the orbitals :math:`\eta_\nu` of the orbital basis given by
    :code:`mol_orbital`,

    .. math::

        \tilde L_{\mu,\alpha\beta} = \langle\omega_\mu | \eta_\alpha\eta_\beta \rangle
        = \int \omega_\mu(r) \eta_\alpha(r) \eta_\beta(r) \mathrm{d} r.

    Args:
        mol_rho: The molecule containing information about the geometry and the density basis set.
        mol_orbital: The molecule containing information about the geometry and the orbital
            basis set.

    Returns:
        Tensor of integrals with shape (n_ao_rho, n_ao_orbital, n_ao_orbital).
    """
    intor = mol_rho._add_suffix("int3c1e")
    # concatenate the environment arrays of both molecules (pyscf wants double indices first)
    atm, bas, env = gto.mole.conc_env(
        mol_orbital._atm,
        mol_orbital._bas,
        mol_orbital._env,
        mol_rho._atm,
        mol_rho._bas,
        mol_rho._env,
    )

    # first and second index run over orbital basis, third over density basis
    shls_slice = (
        0,
        mol_orbital.nbas,
        0,
        mol_orbital.nbas,
        mol_orbital.nbas,
        mol_orbital.nbas + mol_rho.nbas,
    )

    integral = gto.moleintor.getints3c(intor, atm, bas, env, shls_slice)
    return integral.transpose(2, 0, 1)


def get_L2_coulomb_matrix(mol: gto.Mole) -> np.ndarray:
    r"""Compute the two-electron repulsion integral matrix.

    Compute

    .. math::

        D_{\alpha,\beta,\gamma,\delta}  = (\eta^\alpha\eta^\beta|\eta^\gamma\eta^\delta)

    for all basis functions :math:`\eta^\alpha`, :math:`\eta^\beta`, :math:`\eta^\gamma`, and :math:`\eta^\delta`.

    Args:
        mol: The molecule containing information about the geometry and the
            basis set.

    Returns:
        Matrix of integrals.(n_basis, n_basis, n_basis, n_basis)
    """
    intor_name = mol._add_suffix("int2e")
    return gto.moleintor.getints(intor_name, mol._atm, mol._bas, mol._env)


def get_coulomb_matrix(mol: gto.Mole) -> np.ndarray:
    r"""Compute the 2-center Coulomb integral matrix.

    Compute

    .. math::
        \tilde W_{\mu\nu} = \int\int \frac{\omega_\mu(r_1) \omega_\nu(r_2)}{r_{12}} \mathrm{d} r_1
        \mathrm{d} r_2

    for all basis functions :math:`\omega_\mu` and :math:`\omega_\nu`.

    Args:
        mol: The molecule containing information about the geometry and the
            basis set.

    Returns:
        Matrix of integrals.
    """
    intor_name = mol._add_suffix("int2c2e")
    # use that the matrix of integrals is symmetric (hermitian)
    return gto.moleintor.getints(intor_name, mol._atm, mol._bas, mol._env, hermi=1)


def get_coulomb_tensor(
    mol_rho: gto.Mole,
    mol_orbital: gto.Mole,
    shls_slice: tuple | None = None,
    aosym: str = "s1",
) -> np.ndarray:
    r"""Compute the 3-center Coulomb integral tensor between different bases.

    Compute the Coulomb integral over one atomic orbital :math:`\omega_\mu` of the density basis
    given by :code:`mol_rho` and the orbitals :math:`\omega_\nu` of the orbital basis given by
    :code:`mol_orbital`,

    .. math::

        \tilde L_{\mu,\alpha\beta} = (\omega_\mu | \eta_\alpha\eta_\beta)
        = \int\int \frac{\omega_\mu(r_1) \eta_\alpha(r_2) \eta_\beta(r_2)}{r_{12}}
        \mathrm{d} r_1 \mathrm{d} r_2.

    Args:
        mol_rho: The molecule containing information about the geometry and the density basis set.
        mol_orbital: The molecule containing information about the geometry and the orbital
            basis set.

    Returns:
        Tensor of integrals with shape (n_ao_rho, n_ao_orbital, n_ao_orbital).
    """
    intor = mol_rho._add_suffix("int3c2e")
    # concatenate the environment arrays of both molecules (pyscf wants double indices first)
    atm, bas, env = gto.mole.conc_env(
        mol_orbital._atm,
        mol_orbital._bas,
        mol_orbital._env,
        mol_rho._atm,
        mol_rho._bas,
        mol_rho._env,
    )

    # first and second index run over orbital basis, third over density basis
    shls_slice = (
        (
            0,
            mol_orbital.nbas,
            0,
            mol_orbital.nbas,
            mol_orbital.nbas,
            mol_orbital.nbas + mol_rho.nbas,
        )
        if shls_slice is None
        else shls_slice
    )

    integral = gto.moleintor.getints3c(intor, atm, bas, env, shls_slice, aosym=aosym)
    return np.moveaxis(integral, -1, 0)


@lru_cache(maxsize=1)  # Use lru_cache to avoid recomputation during label generation
def get_coulomb_tensor_cached(
    mol_rho: gto.Mole, mol_orbital: gto.Mole, shls_slice: tuple | None = None
) -> np.ndarray:
    return get_coulomb_tensor(mol_rho, mol_orbital, shls_slice)
