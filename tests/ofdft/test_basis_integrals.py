import itertools

import numpy as np
import pytest
import torch
from loguru import logger
from pyscf import dft, gto

from mldft.ofdft.basis_integrals import (
    _append_flat_gaussian,
    _get_one_center_integral,
    get_basis_dipole,
    get_coulomb_matrix,
    get_coulomb_tensor,
    get_gga_potential_matrix,
    get_normalization_vector,
    get_nuclear_attraction_matrix,
    get_nuclear_attraction_vector,
    get_nuclear_gradient_matrix,
    get_nuclear_gradient_vector,
    get_overlap_matrix,
    get_overlap_tensor,
    get_potential_vector,
)
from mldft.utils.molecules import build_mol_with_even_tempered_basis

NUMPY_TO_PYTEST_TOLERANCE = {"rtol": "rel", "atol": "abs"}
# thresholds for comparing analytic and numeric integrals without 1/r on grid
INTEGRATION_THRESHOLD = {"rtol": 1e-5, "atol": 1.5e-5}

# thresholds for comparing analytic and numeric integrals without 1/r on grid
INTEGRATION_THRESHOLD_PYTEST = {
    NUMPY_TO_PYTEST_TOLERANCE[key]: value for key, value in INTEGRATION_THRESHOLD.items()
}

# settings for comparing analytic and numeric integrals with 1/r on grid
INTEGRATION_THRESHOLD_R = {"rtol": 0.01, "atol": 0.15}  # 1/r is numerically less stable
INTEGRATION_THRESHOLD_R_PYTEST = {
    NUMPY_TO_PYTEST_TOLERANCE[key]: value for key, value in INTEGRATION_THRESHOLD_R.items()
}
GRID_LEVEL_R = 2  # use less space to fit distance matrix on GPU
PRUNING_SCHEME_R = dft.treutler_prune  # common dft pruning scheme
test_HCl = [
    gto.M(atom="Cl 0 0 0; H 0 0 1", basis="cc-pvdz"),
    gto.M(atom="Cl 0 0 0; H 0 0 1", basis="pc-3"),
    gto.M(atom="Cl 0 0 0; H 0 0 1", basis="6-31G(2df,p)"),
]
# make cartesian product of HCl molecules
test_molecules_rho_orbital = itertools.product(test_HCl, test_HCl)

np.set_printoptions(precision=4, suppress=True, linewidth=200)

if torch.cuda.is_available():
    GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory
else:
    GPU_MEMORY = 0

REQUIRED_GPU_MEMORY = 14e9


def get_point_charge_potential_on_grid(charges, charges_positions, grid):
    atom_pot = np.zeros_like(grid.weights)
    for Z, coords in zip(charges, charges_positions):
        atom_pot += -Z / np.linalg.norm(grid.coords - coords, axis=1)
    return atom_pot


def get_point_charge_gradient_on_grid(charges, charges_positions, grid):
    atom_pot = np.zeros((charges.size, grid.weights.size, 3))
    for i, (Z, coords) in enumerate(zip(charges, charges_positions)):
        atom_pot[i, ...] += (
            -Z
            * (grid.coords - coords)
            / ((np.linalg.norm(grid.coords - coords, axis=1)) ** 3)[:, None]
        )
    return atom_pot


def test__append_flat_gaussian(molecule_medium):
    """Check dimensions and new env entries."""
    mol = molecule_medium
    bas = mol._bas
    env = mol._env
    bas_mod, env_mod = _append_flat_gaussian(bas, env)
    assert len(bas_mod) == mol.nbas + 1  # one new basis
    assert len(env_mod) == len(mol._env) + 2  # two new coefficients
    assert env_mod[bas_mod[-1, 5]] == 0  # exponent zero
    assert env_mod[bas_mod[-1, 6]] == 1  # coefficient one


def test_get_potential_vector(molecule_ao_grid_level_5):
    """Check dimensions and compare numeric integrals using random potential.

    The integral over the atomic orbitals is computed without einsum and checked. The value of the
    random seed is logged to allow for reproducing the test.
    """
    mol, ao, grid = molecule_ao_grid_level_5
    bit_generator = np.random.PCG64()
    logger.info(f"bit generator state: {bit_generator.state}")
    generator = np.random.Generator(bit_generator)
    potential = generator.standard_normal(grid.weights.shape)

    if ao.ndim > 2:
        ao = ao[0]
    num_integral = get_potential_vector(ao, potential, grid)
    num_integral_2 = (potential * grid.weights) @ ao
    assert num_integral.shape == (mol.nao,)
    assert np.allclose(num_integral_2, num_integral, **INTEGRATION_THRESHOLD)


def test_get_gga_potential_matrix(molecule_ao_grid_level_5):
    """Check dimensions and correct type.

    The value of the random seed is logged to allow for reproducing the test.
    """
    mol, ao, grid = molecule_ao_grid_level_5
    bit_generator = np.random.PCG64()
    logger.info(f"bit generator state: {bit_generator.state}")
    generator = np.random.Generator(bit_generator)
    potential = generator.standard_normal(grid.weights.shape)

    matrix = get_gga_potential_matrix(ao, potential, grid)

    assert matrix.shape == (mol.nao, mol.nao)
    assert matrix.dtype == np.float64
    assert np.all(np.isfinite(matrix))


def test_get_normalization_vector(molecule_ao_grid_level_5):
    """Check dimensions and compare to numeric integral."""
    mol, ao, grid = molecule_ao_grid_level_5
    integral = get_normalization_vector(mol)
    num_integral = get_potential_vector(ao, np.ones_like(grid.weights), grid)
    assert integral.shape == (mol.nao,)
    assert np.allclose(num_integral, integral, **INTEGRATION_THRESHOLD)


def test__get_one_center_integral(molecule_ao_grid_level_5):
    """Check dimensions and compare to numeric integral."""
    mol, ao, grid = molecule_ao_grid_level_5
    integral = _get_one_center_integral(mol, "int1e_ovlp")
    num_integral = get_potential_vector(ao, np.ones_like(grid.weights), grid)
    assert integral.shape == (mol.nao,)
    assert np.allclose(num_integral, integral, **INTEGRATION_THRESHOLD)


def test_get_nuclear_attraction_vector(molecule_ao_grid_level_5):
    """Check dimensions and compare to numeric integral."""
    mol, ao, grid = molecule_ao_grid_level_5
    integral = get_nuclear_attraction_vector(mol)
    if ao.ndim > 2:
        ao = ao[0]

    atom_pot = get_point_charge_potential_on_grid(mol.atom_charges(), mol.atom_coords(), grid)

    num_integral = get_potential_vector(ao, atom_pot, grid)
    assert integral.shape == (mol.nao,)
    assert np.allclose(num_integral, integral, **INTEGRATION_THRESHOLD)


def test_get_nuclear_gradient_vector(molecule_ao_grid_level_5):
    """Check dimensions and compare to numeric integral."""
    mol, ao, grid = molecule_ao_grid_level_5
    integral = get_nuclear_gradient_vector(mol)
    if ao.ndim > 2:
        ao = ao[0]

    atom_pot = get_point_charge_gradient_on_grid(mol.atom_charges(), mol.atom_coords(), grid)

    tmp_num_integral = np.einsum("ni,znx,n->zxi", ao, atom_pot, grid.weights, optimize=True)

    num_integral = tmp_num_integral.reshape(3 * mol.natm, mol.nao, order="C")

    assert integral.shape == (3 * mol.natm, mol.nao)
    assert np.allclose(num_integral, integral, **INTEGRATION_THRESHOLD)


def test_get_nuclear_attraction_matrix(molecule_ao_grid_level_5):
    """Check dimensions, symmetry and compare to numeric integral."""
    mol, ao, grid = molecule_ao_grid_level_5
    integral = get_nuclear_attraction_matrix(mol)
    if ao.ndim > 2:
        ao = ao[0]

    atom_pot = get_point_charge_potential_on_grid(mol.atom_charges(), mol.atom_coords(), grid)

    num_integral = np.einsum("ni,nj,n,n->ij", ao, ao, atom_pot, grid.weights, optimize=True)
    assert integral.shape == (mol.nao, mol.nao)
    assert np.allclose(integral.T, integral)
    assert np.allclose(
        num_integral, integral, **INTEGRATION_THRESHOLD
    ), f"{np.abs(num_integral - integral).max()}"


def test_get_nuclear_gradient_matrix(molecule_ao_grid_level_5):
    """Check dimensions and compare to numeric integral."""
    mol, ao, grid = molecule_ao_grid_level_5
    integral = get_nuclear_gradient_matrix(mol)
    if ao.ndim > 2:
        ao = ao[0]

    atom_pot = get_point_charge_gradient_on_grid(mol.atom_charges(), mol.atom_coords(), grid)

    tmp_num_integral = np.einsum(
        "ni,znx,nj,n->zxij", ao, atom_pot, ao, grid.weights, optimize=True
    )

    num_integral = tmp_num_integral.reshape(3 * mol.natm, mol.nao, mol.nao, order="C")

    assert integral.shape == (3 * mol.natm, mol.nao, mol.nao)
    assert np.allclose(num_integral, integral, **INTEGRATION_THRESHOLD)


def test_get_overlap_matrix(molecule_ao_grid_level_5):
    """Check dimensions, symmetry and compare to numeric integral."""
    mol, ao, grid = molecule_ao_grid_level_5
    integral = get_overlap_matrix(mol)
    if ao.ndim > 2:
        ao = ao[0]
    num_integral = np.einsum("ni,nj,n->ij", ao, ao, grid.weights, optimize=True)
    assert integral.shape == (mol.nao, mol.nao)
    assert np.allclose(integral.T, integral)
    assert np.allclose(num_integral, integral, **INTEGRATION_THRESHOLD)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available.")
@pytest.mark.skipif(
    GPU_MEMORY < REQUIRED_GPU_MEMORY,
    reason=f"Not enough GPU memory available (at least {REQUIRED_GPU_MEMORY/1e9:.1f}GB required,"
    f" {GPU_MEMORY/1e9:.1f}GB available).",
)
def test_get_coulomb_matrix(molecule_medium):
    """Check dimensions, symmetry and compare to numeric integral.

    A comparison to numerical integrals using cuda. Test is only executed if cuda is available with
    at enough memory.
    """
    mol = molecule_medium
    integral = get_coulomb_matrix(mol)
    assert integral.shape == (mol.nao, mol.nao)
    assert np.allclose(integral.T, integral)

    # bigger numerical tests on smaller grid
    grid = dft.Grids(mol)
    grid.level = GRID_LEVEL_R
    grid.prune = PRUNING_SCHEME_R
    grid.build()
    ao = dft.numint.eval_ao(mol, grid.coords, deriv=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # torch.cuda.empty_cache()
    # convert numpy arrays to tensor and move to device
    integral = torch.from_numpy(integral).float().to(device)
    ao = torch.from_numpy(ao).float().to(device)
    coords = torch.from_numpy(grid.coords).float().to(device)
    weights = torch.from_numpy(grid.weights).float().to(device)

    # compute potential using inplace operations to save memory
    potential = torch.linalg.vector_norm(coords[:, None] - coords, dim=2)
    # divide safely
    nonzero_mask = potential != 0
    torch.div(1, potential, out=potential)
    potential[~nonzero_mask] = 0

    potential = torch.einsum("n,nm,m->nm", weights, potential, weights)
    # num_integral = torch.einsum("ni,mj,nm->ij", ao, ao, potential)
    left_contraction = torch.einsum("ni,nm->im", ao, potential)
    num_integral = torch.einsum("im,mj->ij", left_contraction, ao)

    # display used memory
    # logger.info(f"memory allocated: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
    assert torch.allclose(num_integral, integral, **INTEGRATION_THRESHOLD_R)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available.")
@pytest.mark.skipif(
    GPU_MEMORY < REQUIRED_GPU_MEMORY,
    reason=f"Not enough GPU memory available (at least {REQUIRED_GPU_MEMORY/1e9:.1f}GB required),"
    f" {GPU_MEMORY/1e9:.1f}GB available).",
)
@pytest.mark.parametrize("mol_orbital, mol_rho", test_molecules_rho_orbital)
def test_get_coulomb_tensor(mol_orbital: gto.Mole, mol_rho: gto.Mole):
    """Check dimensions, symmetry and compare to numerical integral.

    A comparison to numerical integrals using cuda. Test is only executed if cuda is available with
    at enough memory.
    """
    integral = get_coulomb_tensor(mol_rho, mol_orbital)
    assert integral.shape == (mol_rho.nao, mol_orbital.nao, mol_orbital.nao)
    assert np.allclose(integral.transpose(0, 2, 1), integral)

    # bigger numerical tests on smaller grid
    grid = dft.Grids(mol_rho)
    grid.level = GRID_LEVEL_R
    grid.prune = PRUNING_SCHEME_R
    grid.build()

    ao_rho = dft.numint.eval_ao(mol_rho, grid.coords, deriv=0)
    ao_orbital = dft.numint.eval_ao(mol_orbital, grid.coords, deriv=0)
    ao_orbital_matrix = np.einsum("nj,nk->njk", ao_orbital, ao_orbital)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # convert numpy arrays to tensor and move to device
    integral = torch.from_numpy(integral).float().to(device)
    ao_rho = torch.from_numpy(ao_rho).float().to(device)
    ao_orbital_matrix = torch.from_numpy(ao_orbital_matrix).float().to(device)
    coords = torch.from_numpy(grid.coords).float().to(device)
    weights = torch.from_numpy(grid.weights).float().to(device)

    # compute potential using inplace operations to save memory
    potential = torch.linalg.vector_norm(coords[:, None] - coords, dim=2)
    # divide safely
    nonzero_mask = potential != 0
    torch.div(1, potential, out=potential)
    potential[~nonzero_mask] = 0

    potential = torch.einsum("n,nm,m->nm", weights, potential, weights)
    # num_integral = torch.einsum("ni,mj,nm->ij", ao, ao, potential)
    left_contraction = torch.einsum("ni,nm->im", ao_rho, potential)
    num_integral = torch.einsum("im,mjk->ijk", left_contraction, ao_orbital_matrix)

    # display used memory
    # logger.info(f"memory allocated: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
    assert torch.allclose(num_integral, integral, **INTEGRATION_THRESHOLD_R)


def test_get_overlap_tensor(molecule_small):
    """Check dimensions, symmetry and compare to numerical integral."""
    mol_orbital = molecule_small
    mol_rho = build_mol_with_even_tempered_basis(mol_orbital)
    integral = get_overlap_tensor(mol_rho, mol_orbital)
    assert integral.shape == (mol_rho.nao, mol_orbital.nao, mol_orbital.nao)
    assert np.allclose(integral.transpose(0, 2, 1), integral)

    # bigger numerical tests on smaller grid
    grid = dft.Grids(mol_rho)
    grid.level = GRID_LEVEL_R
    grid.prune = PRUNING_SCHEME_R
    grid.build()

    ao_rho = dft.numint.eval_ao(mol_rho, grid.coords, deriv=0)
    ao_orbital = dft.numint.eval_ao(mol_orbital, grid.coords, deriv=0)

    num_integral = np.einsum(
        "ni,nj,nk,n->ijk", ao_rho, ao_orbital, ao_orbital, grid.weights, optimize=True
    )

    # display used memory
    # logger.info(f"memory allocated: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
    assert (
        pytest.approx(
            num_integral,
            abs=INTEGRATION_THRESHOLD_R["atol"],
            rel=INTEGRATION_THRESHOLD_R["rtol"],
        )
        == integral
    )


def test_get_basis_dipole(molecule_ao_grid_level_5):
    """Check dimensions and compare to numerical integral."""
    mol, ao, grid = molecule_ao_grid_level_5
    integral = get_basis_dipole(mol)
    if ao.ndim > 2:
        ao = ao[0]

    num_integral = np.einsum("gi,gd,g->di", ao, grid.coords, grid.weights, optimize=True)

    assert integral.shape == (3, mol.nao)
    assert np.allclose(num_integral, integral, **INTEGRATION_THRESHOLD)
