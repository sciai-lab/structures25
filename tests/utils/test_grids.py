import numpy as np
import pytest
import torch
from pyscf import dft

from mldft.ofdft import basis_integrals
from mldft.utils.grids import (
    compute_density_density_basis,
    compute_density_orbital_basis,
    get_lebedev_grid,
    get_radial_densities,
)

MEM_SIZES = [10, 100]


@pytest.mark.usefixtures("molecule_medium")
@pytest.mark.parametrize("mem_sizes", MEM_SIZES)
def test_compute_density_orbital_basis(molecule_medium, mem_sizes):
    """Compare blocked implementation against direct matrix vector multiplication."""
    mol = molecule_medium
    grid = dft.Grids(mol)
    grid.level = 3
    grid.build()
    nao = mol.nao
    # Create a random symmetric gamma matrix of shape (nao, nao)
    gamma_np = np.random.rand(nao, nao)
    gamma_np = 0.5 * (gamma_np + gamma_np.T)
    # Evaluate the atomic orbitals on the full grid.
    ao = dft.numint.eval_ao(mol, grid.coords, deriv=0)  # shape: (ngrid, nao)
    # Reference density computed via standard einsum (for each grid point r):
    #   rho(r) = sum_{i,j} ao[r,i] * gamma[i,j] * ao[r,j]
    rho_ref = np.einsum("pi,ij,pj->p", ao, gamma_np, ao)
    max_memory = mem_sizes
    rho_orbital = compute_density_orbital_basis(mol, grid, gamma_np, max_memory=max_memory)
    torch.testing.assert_close(
        rho_orbital,
        torch.as_tensor(rho_ref, dtype=torch.float64),
        rtol=1e-6,
        atol=1e-8,
    )


@pytest.mark.usefixtures("molecule_medium")
@pytest.mark.parametrize("mem_sizes", MEM_SIZES)
def test_compute_density_density_basis(molecule_medium, mem_sizes):
    """Compare blocked implementation against direct matrix vector multiplication."""
    mol = molecule_medium
    grid = dft.Grids(mol)
    grid.level = 3
    grid.build()
    nao = mol.nao
    # Create a random coefficient vector (for the density basis) of length nao.
    coeffs_np = np.random.rand(nao)

    # Evaluate the atomic orbitals on the full grid.
    ao = dft.numint.eval_ao(mol, grid.coords, deriv=0)  # shape: (ngrid, nao)

    # Reference density computed as a simple dot product for each grid point:
    rho_ref = np.dot(ao, coeffs_np)
    max_memory = mem_sizes
    rho_density = compute_density_density_basis(mol, grid, coeffs_np, max_memory=max_memory)

    torch.testing.assert_allclose(
        rho_density,
        torch.tensor(rho_ref, dtype=torch.float64),
        rtol=1e-6,
        atol=1e-8,
    )


def test_get_radial_density(molecule_medium):
    """Test the radial density calculation.

    The main test is to compare the integrated radial density with the number of electrons.
    """
    radii = np.linspace(0, 20, 500)
    center = np.array([0, 0, 0])

    # dumb guess for the coefficients
    normalization_vector = basis_integrals.get_normalization_vector(molecule_medium)
    coeffs = np.zeros(molecule_medium.nao)
    coeffs[normalization_vector > 0] = 1
    coeffs *= molecule_medium.nelectron / (coeffs @ normalization_vector)

    radial_density = get_radial_densities(molecule_medium, center, coeffs, radii, 2030)
    integrated_density = np.trapz(radial_density, radii)

    assert radial_density.shape == (len(radii),)
    assert pytest.approx(integrated_density, rel=2e-5) == molecule_medium.nelectron


def test_get_lebedev_grid():
    """Test the Lebedev grid generation."""
    coords, weights = get_lebedev_grid(590)
    assert coords.shape == (590, 3)
    assert weights.shape == (590,)
    assert pytest.approx(np.sum(weights), rel=1e-10) == 4 * np.pi

    with pytest.raises(AssertionError):
        get_lebedev_grid(1000)
