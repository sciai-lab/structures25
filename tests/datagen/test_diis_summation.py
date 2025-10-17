import numpy as np
import pytest
from pyscf import dft, scf

from mldft.datagen.methods.density_fitting import (
    get_KSDFT_Hartree_potential,
    ksdft_density_matrix,
)
from mldft.datagen.methods.label_generation import diis_weighted_average
from mldft.utils.grids import grid_setup

cycles = 10
basis_functions = 4

arr = np.arange(0, cycles)[:, None] * np.ones((cycles, basis_functions))
ref = np.asarray(
    [
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0, 1.0],
        [1.5, 1.5, 1.5, 1.5],
        [2.0, 2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0, 4.0],
        [5.0, 5.0, 5.0, 5.0],
        [6.0, 6.0, 6.0, 6.0],
        [7.0, 7.0, 7.0, 7.0],
    ]
)


@pytest.mark.parametrize(
    "array, ref",
    [
        (arr, ref),
    ],
)
def test_diis_weighted_average(array: np.ndarray, ref: np.ndarray) -> None:
    """Tests if the diis-function works as intended.

    Just a comparison with hand calculated values
    """
    for i in range(cycles):
        length = min(5, i + 1)
        diis_c = np.ones(length) / length
        assert np.allclose(diis_weighted_average(i, diis_c, array), ref[i])


def test_diis_summation(ksdft_results_small):
    """Checks if the saving and summation of diis works as intended.

    Test to check if:
    - the scraped diis-coefficients are correct
    - the Fock matrix is calculated in the correct way
    - the diis-summation is implemented correctly
    Tested on ksdft_results_all, but reduced to ksdft_results_small to speed test up
    """
    results, initialization, data_of_iteration, mol_with_orbital_basis = ksdft_results_small
    if not results["converged"]:
        return
    # Setting up to calculate the xc functional on the grid
    iterations = len(data_of_iteration)
    xc: str = results["name_xc_functional"].decode()  # strings are stored as bytes in .chk files
    grid_level: int = results["grid_level"]
    prune_method: str = results["prune_method"].decode()
    grid_orbital_basis: dft.Grids = grid_setup(
        mol_with_orbital_basis, grid_level=grid_level, prune=prune_method
    )
    # H_e_kin + H_ext = <eta|T|eta> + <eta|V_ext|eta> kinetic and nuclei-electron-attraction energy hamiltonian (h1e in pyscf)
    kinetic_and_external_energy_hamiltonian: np.ndarray = scf.hf.get_hcore(mol_with_orbital_basis)
    # XC functional
    xc_functional_c: callable(np.ndarray) = lambda gamma: dft.numint.NumInt().nr_rks(
        mol_with_orbital_basis, grid_orbital_basis, xc, gamma
    )
    # overlap matrix
    s1e = mol_with_orbital_basis.intor_symmetric("int1e_ovlp")

    # For the first Fock matrix the initialization density is needed
    first_gamma = initialization["first_density_matrix"]
    # Construction of the first Fock matrix
    kin_ext_pot = kinetic_and_external_energy_hamiltonian
    hartree_pot = get_KSDFT_Hartree_potential(mol_with_orbital_basis, first_gamma)
    n, exchange_correlation_energy_c, xc_pot = xc_functional_c(first_gamma)
    first_fock = kin_ext_pot + hartree_pot + xc_pot  # j-k/2
    first_e_tot = (
        np.einsum("ij,ij", kin_ext_pot + hartree_pot / 2, first_gamma)
        + exchange_correlation_energy_c
        + mol_with_orbital_basis.energy_nuc()
    )
    # Short test if the initial energy can be constructed correctly
    assert first_e_tot == pytest.approx(initialization["first_total_energy"], abs=1e-8)
    mo_coeffs = data_of_iteration[0]["molecular_coeffs_orbitals"]

    # Setting up to test the Fock matrix in every scf iteration
    arr_fock = np.zeros((iterations, mo_coeffs.shape[0], mo_coeffs.shape[0]))
    arr_fock[0, :, :] = first_fock

    for i, data in enumerate(data_of_iteration):
        # load the density, occupation and energy of this iteration
        mo_coeffs = data["molecular_coeffs_orbitals"]
        mo_energy = data["mo_energy"]
        mo_occ = data["occupation_numbers_orbitals"]
        # Construct the Fock matrix from the density and molecular properties
        gamma = ksdft_density_matrix(mo_coeffs, mo_occ)
        kin_ext_pot = kinetic_and_external_energy_hamiltonian
        hartree_pot = get_KSDFT_Hartree_potential(mol_with_orbital_basis, gamma)
        n, exchange_correlation_energy_c, xc_pot = xc_functional_c(gamma)
        fock_matrix = kin_ext_pot + hartree_pot + xc_pot

        # Check if the Fock matrix was calculated correctly
        assert fock_matrix == pytest.approx(data["fock_matrix"], abs=1e-8)

        # This should test if the Fock matrix is calculated in the correct way and if the diis coefficients are correct.
        # The Coefficients and the energies of each iteration are calculated as the eigenvectors and eigenvalues of the
        # diis averaged Fock matrices of the **previous** iterations:
        # F@C = S@C@E
        # where F = sum_i lambda_i F_i is the Fock matrix calculated from the diis-coefficients
        # C is the matrix whose columns are eigenvectors of F, S and E is a diagonal matrix where the entries are the
        # corresponding eigenvalues
        # and the Fock matrices of the previous iterations
        # We now test if
        # C^T@F@C = E
        # C^T@S@C = 1
        # C^T@F@C - C^T@S@C@E =0

        diis_weighted_fock_matrix = diis_weighted_average(
            i, data_of_iteration[i]["diis_coefficients"], arr_fock
        )
        assert mo_coeffs.transpose() @ s1e @ mo_coeffs == pytest.approx(
            np.eye(mo_coeffs.shape[0]), abs=1e-8
        )
        assert mo_coeffs.transpose() @ diis_weighted_fock_matrix @ mo_coeffs == pytest.approx(
            np.diag(mo_energy), abs=1e-8
        )
        # Check if the generalized eigenvalue equation is fulfilled
        assert diis_weighted_fock_matrix @ mo_coeffs - s1e @ mo_coeffs @ np.diag(
            mo_energy
        ) == pytest.approx(np.zeros_like(mo_coeffs), abs=1e-8)
        if i == iterations - 1:
            break
        arr_fock[i + 1, :, :] = fock_matrix
