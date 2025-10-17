"""Provides helper functions for setting up grids for numerical calculations."""


import ctypes
from typing import Callable

import numpy as np
import torch
from pyscf import dft, gto
from pyscf.dft.gen_grid import LEBEDEV_NGRID, libdft

from mldft.utils.coeffs_to_grid import coeffs_to_rho_and_derivatives

STRING_TO_PRUNE = {
    "nwchem_prune": dft.nwchem_prune,
    "treutler_prune": dft.treutler_prune,
    "sg1_prune": dft.sg1_prune,
    "None": None,
}
PRUNE_TO_STRING = {v: k for k, v in STRING_TO_PRUNE.items()}


def grid_setup(
    mol: gto.Mole, grid_level: int = 3, prune: str | Callable | None = "nwchem_prune"
) -> dft.Grids:
    r"""Sets up a grid for numerical calculations of the exchange correlation potential.

    Args:
        mol: the Molecule object in some basis
        grid_level: controls the density of the grid-points
        prune: the pruning scheme

    Returns:
        grid: a grid object

    Raises:
        NotImplementedError : if the specified pruning method is not known.
    """
    if prune is None:
        prune = STRING_TO_PRUNE["None"]
    elif not callable(prune):
        if prune in STRING_TO_PRUNE.keys():
            prune = STRING_TO_PRUNE[prune]
        else:
            raise NotImplementedError(f"The pruning method '{prune}' is not supported.")
    grid = dft.Grids(mol)
    grid.level = grid_level
    grid.prune = prune
    grid.build()
    return grid


def get_lebedev_grid(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    r"""Get Lebedev grid with a given number of points.

    Args:
        n_points: Number of grid points. Allowed values are:
            1, 6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350, 434, 590, 770,
            974, 1202, 1454, 1730, 2030, 2354, 2702, 3074, 3470, 3890, 4334, 4802, 5294, 5810.

    Returns:
        Coordinates and weights of the grid points. The weights sum up to :math:`4 \pi`.
    """
    assert (
        n_points in LEBEDEV_NGRID
    ), f"No Lebedev grid with {n_points} points available. Choose one of \n{LEBEDEV_NGRID}."

    lebedev_grid = np.empty((n_points, 4))
    libdft.MakeAngularGrid(lebedev_grid.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(n_points))
    coords = lebedev_grid[:, :3]
    weights = 4 * np.pi * lebedev_grid[:, 3]

    return coords, weights


def get_radial_densities(
    mol: gto.Mole,
    center: np.ndarray,
    coeffs: np.ndarray,
    radii: np.ndarray,
    n_spherical_points: int,
) -> np.ndarray:
    """Get radial density for a given number of points and radii.

    Args:
        mol: The molecule with density basis.
        center: Center of the radial density.
        coeffs: Coefficients of the basis functions. If the dimension is one, they are interpreted
            as a single set of coefficients. If the dimension is two, they are interpreted as a
            list of coefficients with shape (n_densities, n_basis_functions).
            The coefficients are assumed to be untransformed.
        radii: Radii of the spheres in Bohr.
        n_spherical_points: Number of grid points on the sphere. Possible values are listed in
            :py:func:`get_lebedev_grid`.

    Returns:
        Radial densities at given radii.
    """
    n_radii = len(radii)
    coords_sphere, weights = get_lebedev_grid(n_spherical_points)

    coords = np.einsum("i,jk->ijk", radii, coords_sphere)
    coords = coords + center
    coords = coords.reshape(n_radii * n_spherical_points, 3)

    ao = mol.eval_gto("GTOval", coords)
    ao = ao.reshape(n_radii, n_spherical_points, mol.nao)
    ao_integrated = np.einsum("rgi,g->ri", ao, weights)

    radial_density = np.einsum("ri,...i->...r", ao_integrated, coeffs)
    radial_density = radial_density * radii**2  # from the Jacobian determinant

    return radial_density


def compute_max_block_size(
    max_memory: float,
    num_aos: int,
    heuristic_multiplier: int = 8,
) -> int:
    """Compute the maximum number of grid points per block.

    The estimated memory usage per grid point is assumed to be:
        memory_usage_per_point = dtype_bytes * heuristic_multiplier * num_aos
    where:
        - dtype_bytes: the number of bytes per tensor entry (e.g., 8 for float64)
        - heuristic_multiplier: an empirical factor that estimates the additional memory
          overhead per grid point when computing the XC functional.

    The total memory allowed is given in MB, so we first convert it to bytes, then compute:
        max_block_size = floor((max_memory_in_bytes) / memory_usage_per_point)

    Args:
        max_memory: Maximum memory (in MB) allowed.
                              when computing the XC functional.
        num_aos: Number of atomic orbitals (or related dimension).
        heuristic_multiplier: A heuristic factor estimating the actual memory usage per grid point entry

    Returns:
        Maximum number of grid points in one block.
    """
    mega_byte = 1048576  # number of bytes in 1 MB
    return int(np.floor(mega_byte * max_memory / (heuristic_multiplier * 8 * num_aos)))


def get_grid_blocks(grid_size: int, block_size: int) -> list[tuple[int, int]]:
    """Split the grid into blocks defined by start and end indices.

    Args:
        grid_size: Total number of grid points.
        block_size: Maximum number of grid points per block.

    Returns:
        List of tuples (block_start, block_end) for each block.
    """
    return [(i, min(i + block_size, grid_size)) for i in range(0, grid_size, block_size)]


def compute_density_orbital_basis(
    mol_with_orbital_basis: gto.Mole,
    grid: dft.Grids,
    gamma: np.ndarray | torch.Tensor,
    max_memory: float = 4000.0,
):
    """Compute the density on the grid using the orbital basis."""
    if isinstance(gamma, np.ndarray):
        gamma = torch.as_tensor(gamma, dtype=torch.float64)

    max_block_size = compute_max_block_size(
        max_memory, mol_with_orbital_basis.nao, heuristic_multiplier=8
    )
    grid_chunks = get_grid_blocks(grid.size, max_block_size)
    rho = torch.zeros(grid.size, dtype=torch.float64)

    for block_start, block_end in grid_chunks:
        ao_block = dft.numint.eval_ao(
            mol_with_orbital_basis, grid.coords[block_start:block_end], deriv=0
        )
        rho_orbital = dft.numint.eval_rho(mol_with_orbital_basis, ao_block, dm=gamma, xctype="LDA")
        rho[block_start:block_end] = torch.as_tensor(rho_orbital, dtype=torch.float64)

    return rho


def compute_density_density_basis(
    mol_with_density_basis: gto.Mole,
    grid: dft.Grids,
    coeffs: np.ndarray | torch.Tensor,
    max_memory: float = 4000.0,
):
    """Compute the density on the grid using the density basis."""
    if isinstance(coeffs, np.ndarray):
        coeffs = torch.as_tensor(coeffs, dtype=torch.float64)

    max_block_size = compute_max_block_size(
        max_memory, mol_with_density_basis.nao, heuristic_multiplier=4
    )
    grid_chunks = get_grid_blocks(grid.size, max_block_size)
    rho = torch.zeros(grid.size, dtype=torch.float64)

    for block_start, block_end in grid_chunks:
        ao_block = dft.numint.eval_ao(
            mol_with_density_basis, grid.coords[block_start:block_end], deriv=0
        )
        ao_block = torch.as_tensor(ao_block, dtype=torch.float64)
        rho_density = coeffs_to_rho_and_derivatives(coeffs, ao_block, max_derivative_order=0)[0]
        rho[block_start:block_end] = rho_density

    return rho


def compute_l1_norm_orbital_vs_density_basis(
    mol_with_orbital_basis: gto.Mole,
    mol_with_density_basis: gto.Mole,
    grid: dft.Grids,
    gamma: np.ndarray,
    coeffs: np.ndarray,
    max_memory: float = 4000.0,
):
    """Compute the L1 norm (sum of absolute differences) between the densities computed in the
    orbital and density basis.

    This function computes the density over the same grid using two different approaches
    (orbital and density bases) and then returns the L1 norm of the difference.

    Args:
        mol_with_orbital_basis: Molecule object using the orbital basis.
        mol_with_density_basis: Molecule object using the density basis.
        grid: Grid on which the densities are evaluated.
        gamma: Density matrix used in the orbital basis evaluation.
        coeffs: Coefficients used in the density basis evaluation.
        max_memory: Maximum memory (in MB) allowed for processing grid blocks.

    Returns:
        L1 norm (a torch scalar) of the density difference.
    """
    rho_orbital = compute_density_orbital_basis(mol_with_orbital_basis, grid, gamma, max_memory)
    rho_density = compute_density_density_basis(mol_with_density_basis, grid, coeffs, max_memory)
    grid_weights = torch.as_tensor(grid.weights, dtype=torch.float64)
    l1_norm_difference = torch.dot(torch.abs(rho_orbital - rho_density), grid_weights)
    return l1_norm_difference
