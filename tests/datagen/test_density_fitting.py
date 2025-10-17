import numpy as np
import pytest
from pyscf import dft

from mldft.datagen.methods.density_fitting import ksdft_density_matrix
from mldft.datagen.methods.label_generation import calculate_labels
from mldft.ofdft.basis_integrals import (
    get_coulomb_matrix,
    get_coulomb_tensor,
    get_L2_coulomb_matrix,
    get_normalization_vector,
    get_nuclear_attraction_matrix,
    get_nuclear_attraction_vector,
    get_overlap_matrix,
    get_overlap_tensor,
)
from mldft.utils.grids import grid_setup
from mldft.utils.molecules import construct_aux_mol

ans_orbitals = 10
ans_basis_functions = 10
ans_density_basis_functions = 16
C = np.random.normal(size=(ans_basis_functions, ans_orbitals))
gamma = C @ C.T
temp2 = np.random.normal(size=(ans_density_basis_functions, ans_density_basis_functions))
W_coulumb = np.diag(np.ones(ans_density_basis_functions)) + (temp2 + temp2.T) * 0.1
temp = np.random.normal(
    size=(ans_density_basis_functions, ans_basis_functions, ans_basis_functions)
)
L_coulumb = temp + np.swapaxes(temp, 1, 2)
v_ext_p = np.random.normal(size=(ans_density_basis_functions))
temp2 = np.random.normal(size=(ans_basis_functions, ans_basis_functions))
V_ext_C = temp2 + temp2.T

test_matricies = [(W_coulumb, L_coulumb, gamma, v_ext_p, V_ext_C)]


def sum_hartree_external_energy_of_residual_density(
    W_coulomb: np.ndarray,
    L_coulomb: np.ndarray,
    gamma: np.ndarray,
    v_ext_p: np.ndarray,
    V_ext_C: np.ndarray,
    p: np.ndarray,
) -> np.floating:
    """This calculates the sum of hartree and external energy of the external energy minus the
    hartree energy of the ks-density.

    Args:
        L_overlap: 3-center coulomb matrix of the overlap of ks- and of-coefficients
        W_overlap: 2-center coulomb matrix of the of-coefficients
        v_external_p: 1 center external potential vector of the of-coefficients
        v_external_C: 1 center external potential matrix of the ks-coefficients

    Returns:
        The energy
    """
    return (
        p.T @ W_coulomb @ p
        - 2 * p.T @ np.einsum("ijk,jk->i", L_coulomb, gamma, optimize=True)
        + (p.T @ v_ext_p - np.trace(gamma @ V_ext_C)) ** 2
    )


def l2_norm_residual_density(
    W_overlap: np.ndarray,
    L_overlap: np.ndarray,
    l2_norms_orbital_density: np.ndarray,
    p: np.ndarray,
    gamma: np.ndarray,
):
    """This calculates the L2 norm of the residual density.

    Args:
        L_overlap: 3-center coulomb matrix of the overlap of ks- and of-coefficients
        W_overlap: 2-center coulomb matrix of the of-coefficients
        l2_norms_orbital_density: the L2 norm of the orbital density
        p: vector of the of-coefficients
        gamma: matrix of the ks-coefficients

    Returns:
        The L2 norm of the residual density
    """
    pWp = np.einsum("i,ij,j->", p, W_overlap, p)
    pLgamma = np.einsum("ijk,jk,i->", L_overlap, gamma, p, optimize=True)
    return pWp - 2 * pLgamma + l2_norms_orbital_density


def l2_norm_density_orbitals(gamma: np.ndarray, ao: np.ndarray, grid):
    """Calculates the L2 norm of the density in the orbital basis on the grid.

    (because pyscf doesn`t support int4c1e)

    Args:
        gamma: coefficients of the ks-density
        ao: orbital basis functions evaluated on the gridpoints
        grid: grid object

    Returns:
        The L2 norm of the ks-density
    """
    rho_on_grid = np.einsum("ij,jk,ik->i", ao, gamma, ao)
    rho2_on_grid = rho_on_grid**2
    l2_norm = rho2_on_grid @ grid.weights
    return l2_norm


def density_fitting_test_for_real_molecules(
    ksdft_results: tuple[dict, list[dict]], print_results=True
):
    """Test if the density fit function provides a good fit to the original orbitals."""

    results, initialization, data_of_iteration, mol_with_orbital_basis = ksdft_results
    aux_basis_names = [
        "even_tempered_2.5",
        "even_tempered_2.0",
    ]  # ,"even_tempered_1.5", "def2-universal-jfit"]
    metric_names = [
        "hartree_energy",
        "exchange_correlation_energy",
        "external_energy",
        "kinetic_energy",
        "sum_hartree_external",
        "total_density_difference",
        "Norms",
        "L2_norm_residual_density",
    ]
    methods = (
        "hartree",
        "hartree+external",
        "hartree+external_mofdft",
        "hartree+external_mofdft_enforced_density",
        "hartree+external_mofdft_fixed_density",
        "hartree+external_fixed_density",
        "overlap",
        "hartree_fixed_density_external",
    )
    metric_values = np.zeros(
        (len(aux_basis_names), len(methods), len(metric_names), len(data_of_iteration) + 1)
    )
    for a, aux_basis_name in enumerate(aux_basis_names):
        mol_with_density_basis = construct_aux_mol(mol_with_orbital_basis, aux_basis_name, "Bohr")

        # KS uses for this basis in densityfitting an even tempered basis with beta = 2.0
        # if not results["converged"]:
        #    return
        iterations = len(data_of_iteration)
        print(
            f"orbitalbasis:\t{mol_with_orbital_basis.basis}\t{mol_with_orbital_basis.nao}\tdensitybasis:\t{aux_basis_name}\t{mol_with_density_basis.nao}"
        )

        w_coulomb = np.asarray(get_coulomb_matrix(mol_with_density_basis))
        l_coulomb = np.asarray(get_coulomb_tensor(mol_with_density_basis, mol_with_orbital_basis))
        d_coulomb = np.asarray(get_L2_coulomb_matrix(mol_with_orbital_basis))
        w_overlap = np.asarray(get_overlap_matrix(mol_with_density_basis))
        l_overlap = np.asarray(get_overlap_tensor(mol_with_density_basis, mol_with_orbital_basis))

        w_basis_integrals = np.asarray(get_normalization_vector(mol_with_density_basis))
        orbitals_basis_integrals = np.asarray(get_overlap_matrix(mol_with_orbital_basis))

        grid_orbitals = grid_setup(mol_with_orbital_basis, grid_level=results["grid_level"])
        ao = dft.numint.eval_ao(mol_with_orbital_basis, grid_orbitals.coords, deriv=0)

        external_potential_p = np.asarray(get_nuclear_attraction_vector(mol_with_density_basis))
        external_potential_c = np.asarray(get_nuclear_attraction_matrix(mol_with_orbital_basis))

        def get_norm(gamma, p):
            return (
                p.T @ w_coulomb @ p
                - 2 * p.T @ np.einsum("ijk,jk->i", l_coulomb, gamma, optimize=True)
                + np.einsum("ij,ijkl,kl->", gamma, d_coulomb, gamma, optimize=True)
                + (p.T @ external_potential_p - np.trace(gamma @ external_potential_c)) ** 2
            )

        gammas = np.stack(
            [initialization["first_density_matrix"]]
            + [
                ksdft_density_matrix(
                    data["molecular_coeffs_orbitals"], data["occupation_numbers_orbitals"]
                )
                for data in data_of_iteration
            ]
        )

        label_dicts = [
            calculate_labels(
                results,
                initialization,
                data_of_iteration,
                mol_with_orbital_basis,
                mol_with_density_basis,
                density_fitting_method=method,
            )
            for method in methods
        ]

        ks_label_names = ["ks_e_hartree", "ks_e_xc", "ks_e_ext", "ks_e_kin"]
        of_label_names = ["of_e_hartree", "of_e_xc", "of_e_ext", "of_e_kin"]

        of_coeffs = np.stack([label_dict["of_coeffs"] for label_dict in label_dicts])

        ks_label_energies = np.stack([label_dicts[0][label_name] for label_name in ks_label_names])
        hartree_ext = (ks_label_energies[0] + ks_label_energies[2])[None, :]
        ks_label_energies = np.concatenate(
            (ks_label_energies, hartree_ext)
        )  # ks_hartree_energy + ks_external_energy
        ks_densitys_orbitals = np.einsum("ij,kij->k", orbitals_basis_integrals, gammas)[None, :]
        ks_label_energies = np.concatenate((ks_label_energies, ks_densitys_orbitals))

        of_label_energies = np.stack(
            [
                np.stack([label_dict[label_name] for label_name in of_label_names])
                for label_dict in label_dicts
            ]
        )
        of_label_energies = np.concatenate(
            (of_label_energies, (of_label_energies[:, 0] + of_label_energies[:, 2])[:, None, :]),
            axis=1,
        )  # of_hartree_energy + of_external_energy
        of_densitys = np.einsum("i,jki->jk", w_basis_integrals, of_coeffs)[:, None, :]
        of_label_energies = np.concatenate(
            (of_label_energies, of_densitys),
            axis=1,
        )

        ae_energies = np.abs(ks_label_energies - of_label_energies)

        norms = np.asarray(
            [
                [get_norm(gammas[i], of_coeffs[j][i]) for i in range(iterations + 1)]
                for j in range(len(methods))
            ]
        )

        l2_norms_orbital_density = np.asarray(
            [l2_norm_density_orbitals(gammas[i], ao, grid_orbitals) for i in range(iterations + 1)]
        )

        L2_norms_residual_density = np.asarray(
            [
                [
                    l2_norm_residual_density(
                        w_overlap,
                        l_overlap,
                        l2_norms_orbital_density[i],
                        of_coeffs[j][i],
                        gammas[i],
                    )
                    for i in range(iterations + 1)
                ]
                for j in range(len(methods))
            ]
        )
        metric_values[a] = np.concatenate(
            (ae_energies, norms[:, None, :], L2_norms_residual_density[:, None, :]), axis=1
        )
    assert pytest.approx(metric_values, 1.0e3) == np.zeros_like(metric_values)
    if print_results:
        max_error = np.max(metric_values, axis=3)
        best_index = np.argmin(metric_values, axis=1)
        best_index_overall = np.argmin(max_error, axis=1)

        best_value_over_methods = np.min(max_error, axis=1)
        best_index_over_methods = np.argmin(best_value_over_methods, axis=0)
        for a, aux_basis_name in enumerate(aux_basis_names):
            empty = " "
            # for cycle in range(iterations):
            #    print(f"Cycle {cycle:20}    " + "".join([f"{method:20}" for method in methods]) + "\n")
            #    for i, name in enumerate(metric_names):
            #       print(f"{name:30}" + "".join([f"{val:4.2e}, {empty:11}" for val in metric_values[a,:, i, cycle]]) + f"\t{methods[best_index[a,i, cycle]]}")

            print(f"\n The best method for each metric is for basisname: {aux_basis_name}\n")
            print(f"{empty:30}" + "".join([f"{method:20}" for method in methods]) + "\n")
            for i, name in enumerate(metric_names):
                print(
                    f"{name:30}"
                    + "".join([f"{val:4.2e}{empty:12}" for val in max_error[a, :, i]])
                    + f"\t{methods[best_index_overall[a, i]]}"
                )
        print(
            "basis_name:\t\t\t\t"
            + "".join([f"{aux_basis_name:20}" for aux_basis_name in aux_basis_names])
            + "\n"
        )
        for i, name in enumerate(metric_names):
            print(
                f"{name:30}"
                + "".join([f"{val:4.2e}\t" for val in best_value_over_methods[:, i]])
                + f"\t{aux_basis_names[best_index_over_methods[i]]}"
            )


@pytest.mark.slow
def test_density_fitting_for_real_molecules(ksdft_results_small):
    """Test if the density fit function provides a good fit to the original orbitals."""
    density_fitting_test_for_real_molecules(ksdft_results_small)
