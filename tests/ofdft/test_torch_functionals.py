import numpy as np
import opt_einsum
import pytest
import torch
from pyscf import dft
from pyscf.dft import libxc, numint

from mldft.ofdft.libxc_functionals import eval_libxc_functionals, required_derivative
from mldft.ofdft.torch_functionals import (
    eval_torch_functionals_blocked,
    eval_torch_functionals_blocked_fast,
    torch_functional,
)
from mldft.utils.coeffs_to_grid import coeffs_to_rho_and_derivatives
from mldft.utils.grids import grid_setup
from mldft.utils.molecules import build_mol_with_even_tempered_basis
from mldft.utils.utils import set_default_torch_dtype

# Can be extended later
gga_list = [("GGA_K_APBE", "cpu"), ("GGA_C_PBE", "cpu"), ("GGA_X_PBE", "cpu")]
if torch.cuda.is_available():
    gga_list += [(functional_string, "cuda") for functional_string, _ in gga_list]

functionals = [["GGA_K_APBE"], ["GGA_C_PBE"], ["GGA_X_PBE"]]
max_memory_list = [1, 4000]
functional_memory = [
    (functional, memory) for functional in functionals for memory in max_memory_list
]


@set_default_torch_dtype(torch.float64)
@pytest.mark.parametrize("functional_string, device", gga_list)
def test_torch_functional(functional_string, device, molecule_small):
    """Test the torch implementation for the functionals."""
    torch.autograd.set_detect_anomaly(True)
    mol = molecule_small
    grid = dft.Grids(mol)
    grid.level = 3
    grid.build()
    ao_value = numint.eval_ao(mol, grid.coords, deriv=1)
    dm = np.random.random((mol.nao_nr(), mol.nao_nr()))
    dm = dm + dm.T
    rho = numint.eval_rho(mol, ao_value, dm, xctype="GGA")
    ex, vxc, fxc, kxc = libxc.eval_xc(functional_string, rho, deriv=1)
    # vxc[0] functional derivative wrt rho
    # vxc[1] functional derivative wrt |grad rho @ grad rho|
    libxc_energy_tensor = torch.tensor(ex)
    # Torch part
    if functional_string == "GGA_K_APBE":
        functional_string = "APBE"
    (
        kinetic_energy_density,
        kinetic_energy_density_gradient_rho,
        kinetic_energy_density_gradient_s,
    ) = torch_functional(
        torch.tensor(rho, requires_grad=True, device=device), functional=functional_string
    )
    # Assert total energy, energy density and energy density gradient are equal
    torch.testing.assert_close(libxc_energy_tensor.sum(), kinetic_energy_density.cpu().sum())
    torch.testing.assert_close(libxc_energy_tensor, kinetic_energy_density.cpu())
    torch.testing.assert_close(torch.tensor(vxc[0]), kinetic_energy_density_gradient_rho.cpu())
    torch.testing.assert_close(torch.tensor(vxc[1]), kinetic_energy_density_gradient_s.cpu())


@set_default_torch_dtype(torch.float64)
@pytest.mark.parametrize("functional_string, device", gga_list)
def test_torch_functional_with_coeffs(functional_string, device, molecule_medium):
    """Test torch implementation for densities given in coeffs."""
    mol_with_orbital_basis = molecule_medium
    mol_with_density_basis = build_mol_with_even_tempered_basis(mol_with_orbital_basis)
    grid_level = 3
    prune_method = "nwchem_prune"
    grid_density_basis: dft.Grids = grid_setup(
        mol_with_density_basis, grid_level=grid_level, prune=prune_method
    )
    p = np.random.random((mol_with_density_basis.nao_nr(),))
    p = p / np.linalg.norm(p)

    max_derivative = required_derivative(functional_string)
    ao = dft.numint.eval_ao(
        mol_with_density_basis, grid_density_basis.coords, deriv=max_derivative
    )
    kinetic_energy_p, kinetic_potential_p = eval_libxc_functionals(
        p, functional_string, grid_density_basis, ao
    )

    # Torch part
    p_torch = torch.tensor(p, dtype=torch.float64, requires_grad=True, device=device)
    ao_torch = torch.tensor(ao, dtype=torch.float64, requires_grad=True, device=device)
    rho_gga = coeffs_to_rho_and_derivatives(p_torch, ao_torch, 1)
    if functional_string == "GGA_K_APBE":
        functional_string = "APBE"
    energy_torch_grid, _, _ = torch_functional(
        rho_gga, functional=functional_string, get_gradient=False
    )
    kin_energy_torch = opt_einsum.contract(
        "i,i,i->",
        rho_gga[0],
        energy_torch_grid,
        torch.tensor(grid_density_basis.weights, dtype=torch.float64, device=device),
    )

    grad_torch = torch.autograd.grad(kin_energy_torch, p_torch)[0]
    torch.testing.assert_close(
        kin_energy_torch.cpu(), torch.as_tensor(kinetic_energy_p, dtype=torch.float64)
    )
    torch.testing.assert_close(
        grad_torch.cpu(), torch.as_tensor(kinetic_potential_p, dtype=torch.float64)
    )


@set_default_torch_dtype(torch.float64)
@pytest.mark.parametrize("functional_list,max_memory", functional_memory)
def test_blocked_torch_functionals(functional_list, molecule_medium, max_memory):
    """Test torch implementation for densities given in coeffs."""
    assert len(functional_list) == 1, "Test is only designed for one functional at a time."
    mol_with_orbital_basis = molecule_medium
    mol_with_density_basis = build_mol_with_even_tempered_basis(mol_with_orbital_basis)
    grid_level = 3
    prune_method = "nwchem_prune"
    grid_density_basis: dft.Grids = grid_setup(
        mol_with_density_basis, grid_level=grid_level, prune=prune_method
    )
    p = np.random.random((mol_with_density_basis.nao_nr(),))
    p = p / np.linalg.norm(p)
    max_derivative = required_derivative(functional_list)
    ao = dft.numint.eval_ao(
        mol_with_density_basis, grid_density_basis.coords, deriv=max_derivative
    )
    kinetic_energy_p, kinetic_potential_p = eval_libxc_functionals(
        p, functional_list, grid_density_basis, ao
    )
    kinetic_energy_libxc = torch.as_tensor(kinetic_energy_p[0], dtype=torch.float64)
    kinetic_potential_libxc = torch.as_tensor(kinetic_potential_p, dtype=torch.float64)
    # Torch part
    p_torch = torch.tensor(p, dtype=torch.float64, requires_grad=True)
    energies_and_gradients = eval_torch_functionals_blocked(
        mol_with_density_basis, grid_density_basis, p_torch, functional_list, max_memory=max_memory
    )
    functional = functional_list[0]
    energy_torch, grad_torch = energies_and_gradients[functional]
    torch.testing.assert_close(kinetic_energy_libxc, energy_torch)
    torch.testing.assert_close(kinetic_potential_libxc, grad_torch)


@set_default_torch_dtype(torch.float64)
@pytest.mark.parametrize("functional_list,max_memory", functional_memory)
def test_blocked_torch_functionals_fast(functional_list, molecule_medium, max_memory):
    """Test torch implementation for densities given in coeffs."""
    assert len(functional_list) == 1, "Test is only designed for one functional at a time."
    mol_with_orbital_basis = molecule_medium
    mol_with_density_basis = build_mol_with_even_tempered_basis(mol_with_orbital_basis)
    grid_level = 3
    prune_method = "nwchem_prune"
    grid_density_basis: dft.Grids = grid_setup(
        mol_with_density_basis, grid_level=grid_level, prune=prune_method
    )
    p = np.random.random((mol_with_density_basis.nao_nr(),))
    p = p / np.linalg.norm(p)
    max_derivative = required_derivative(functional_list)
    ao = dft.numint.eval_ao(
        mol_with_density_basis, grid_density_basis.coords, deriv=max_derivative
    )
    kinetic_energy_p, kinetic_potential_p = eval_libxc_functionals(
        p, functional_list, grid_density_basis, ao
    )
    kinetic_energy_libxc = torch.as_tensor(kinetic_energy_p[0], dtype=torch.float64)
    kinetic_potential_libxc = torch.as_tensor(kinetic_potential_p, dtype=torch.float64)
    # Torch part
    p_torch = torch.tensor(p, dtype=torch.float64, requires_grad=True)
    ao_torch = torch.tensor(ao, dtype=torch.float64)
    grid_weights_torch = torch.tensor(grid_density_basis.weights, dtype=torch.float64)
    energies_and_gradients = eval_torch_functionals_blocked_fast(
        ao_torch, grid_weights_torch, p_torch, functional_list, max_memory=max_memory
    )
    functional = functional_list[0]
    energy_torch, grad_torch = energies_and_gradients[functional]
    torch.testing.assert_close(kinetic_energy_libxc, energy_torch)
    torch.testing.assert_close(kinetic_potential_libxc, grad_torch)
