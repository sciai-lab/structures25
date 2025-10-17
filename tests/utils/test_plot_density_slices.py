import numpy as np
import pytest

from mldft.datagen.methods.density_fitting import (
    get_density_fitting_function,
    ksdft_density_matrix,
)
from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.of_data import OFData
from mldft.ofdft.basis_integrals import (
    get_coulomb_matrix,
    get_coulomb_tensor,
    get_nuclear_attraction_matrix,
    get_nuclear_attraction_vector,
)
from mldft.utils.molecules import construct_aux_mol
from mldft.utils.plotting.plot_density_slices import PlotDensitySlices


@pytest.fixture()
def of_coeffs_fitted_small(ksdft_results_small):
    results, initialization, data_of_iteration, mol = ksdft_results_small
    mo_coeff = data_of_iteration[-1]["molecular_coeffs_orbitals"]
    mo_occ = data_of_iteration[-1]["occupation_numbers_orbitals"]
    gamma = ksdft_density_matrix(mo_coeff, mo_occ)
    basis_set = "even_tempered_2.5"
    density_fitting_method = "overlap"
    # Build molecules for each basis set
    aux_mol = construct_aux_mol(mol, aux_basis_name=basis_set)
    aux_mol.build()
    # build list of of-coeffs for all df-methods for this basis set
    W_coulomb = np.asarray(get_coulomb_matrix(aux_mol))
    L_coulomb = get_coulomb_tensor(aux_mol, mol)
    external_potential_p = np.asarray(get_nuclear_attraction_vector(aux_mol))
    external_potential_c = np.asarray(get_nuclear_attraction_matrix(mol))

    of_coeffs = get_density_fitting_function(
        density_fitting_method,
        mol,
        aux_mol,
        W_coulomb,
        L_coulomb,
        external_potential_c,
        external_potential_p,
    )(gamma)
    return of_coeffs, aux_mol, mol


@pytest.fixture()
def of_coeffs_fitted_small_different_basis_sets(ksdft_results_small):
    results, initialization, data_of_iteration, mol = ksdft_results_small
    mo_coeff = data_of_iteration[-1]["molecular_coeffs_orbitals"]
    mo_occ = data_of_iteration[-1]["occupation_numbers_orbitals"]
    gamma = ksdft_density_matrix(mo_coeff, mo_occ)
    basis_sets = ["even_tempered_2.5", "even_tempered_5.0"]
    of_coeffs_list = []
    aux_mol_list = []
    for basis_set in basis_sets:
        density_fitting_method = "overlap"
        # Build molecules for each basis set
        aux_mol = construct_aux_mol(mol, aux_basis_name=basis_set)
        aux_mol.build()
        # build list of of-coeffs for all df-methods for this basis set
        W_coulomb = np.asarray(get_coulomb_matrix(aux_mol))
        L_coulomb = get_coulomb_tensor(aux_mol, mol)
        external_potential_p = np.asarray(get_nuclear_attraction_vector(aux_mol))
        external_potential_c = np.asarray(get_nuclear_attraction_matrix(mol))

        of_coeffs = get_density_fitting_function(
            density_fitting_method,
            mol,
            aux_mol,
            W_coulomb,
            L_coulomb,
            external_potential_c,
            external_potential_p,
        )(gamma)
        of_coeffs_list.append(of_coeffs)
        aux_mol_list.append(aux_mol)
    return of_coeffs_list, aux_mol_list, mol


@pytest.fixture()
def different_kohn_sham_coeffs(ksdft_results_small):
    results, initialization, data_of_iteration, mol = ksdft_results_small
    mo_coeff_0 = data_of_iteration[0]["molecular_coeffs_orbitals"]
    mo_occ_0 = data_of_iteration[0]["occupation_numbers_orbitals"]
    gamma_0 = ksdft_density_matrix(mo_coeff_0, mo_occ_0)
    mo_coeff_1 = data_of_iteration[-1]["molecular_coeffs_orbitals"]
    mo_occ_1 = data_of_iteration[-1]["occupation_numbers_orbitals"]
    gamma_1 = ksdft_density_matrix(mo_coeff_1, mo_occ_1)
    return mol, gamma_1, mol, gamma_0


@pytest.fixture()
def of_data_sample_small(of_coeffs_fitted_small):
    of_coeffs, aux_mol, mol = of_coeffs_fitted_small
    basis_info = BasisInfo.from_mol(aux_mol)
    return OFData.construct_new(
        basis_info,
        aux_mol.atom_coords(),
        aux_mol.atom_charges(),
        of_coeffs,
        scf_iteration=-1,
        ks_basis=mol.basis,
    )


@pytest.mark.skip()
def test_show_molecules(molecule_all):
    pds = PlotDensitySlices.show_mol_atoms(molecule_all)
    pds.plot_difference_interactive()


@pytest.mark.skip()
def test_show_molecules_ks_density(molecule_all):
    pds = PlotDensitySlices.show_mol_ks_density(molecule_all)
    pds.plot_difference_interactive()


@pytest.mark.skip()
def test_compare_kohn_sham_densities(different_kohn_sham_coeffs):
    mol_0, gamma_0, mol_1, gamma_1 = different_kohn_sham_coeffs
    pds = PlotDensitySlices.show_difference_orbital_densities(
        mol_0, gamma_0, [mol_1, mol_0], [gamma_1, gamma_0], ["first_iteration", "other"]
    )
    pds.plot_difference_interactive()


@pytest.mark.skip()
def test_compare_of_densities(of_coeffs_fitted_small_different_basis_sets):
    coeffs_list, aux_mol_list, _ = of_coeffs_fitted_small_different_basis_sets
    pds = PlotDensitySlices.show_difference_of_densities(
        aux_mol_list[0],
        coeffs_list[0],
        aux_mol_list,
        coeffs_list,
        ["even_tempered_2.5", "even_tempered_5.0"],
    )
    pds.plot_difference_interactive()


@pytest.mark.skip()
def test_show_molecules_ks_density_given(ksdft_results_small):
    results, initialization, data_of_iteration, mol_with_orbital_basis = ksdft_results_small
    mo_coeff = data_of_iteration[-1]["molecular_coeffs_orbitals"]
    mo_occ = data_of_iteration[-1]["occupation_numbers_orbitals"]
    gamma = ksdft_density_matrix(mo_coeff, mo_occ)
    pds = PlotDensitySlices.show_mol_ks_density(mol_with_orbital_basis, gamma)
    pds.plot_difference_interactive()


@pytest.mark.skip()
def test_show_molecules_ks_and_of_densities(molecule_small):
    pds = PlotDensitySlices.show_mol_ks_and_of_density(molecule_small)
    pds.plot_difference_interactive()


@pytest.mark.skip()
def test_show_molecules_of_densities(of_coeffs_fitted_small):
    of_coeffs, auxmol, orbital_mol = of_coeffs_fitted_small
    pds = PlotDensitySlices.show_mol_of_density(of_coeffs, auxmol)
    pds.plot_difference_interactive()


@pytest.mark.skip()
def test_show_of_data(of_data_sample_small):
    pds = PlotDensitySlices.show_of_data(of_data_sample_small)
    pds.plot_difference_interactive()


@pytest.mark.skip()
def test_show_of_data_vs_ks(of_data_sample_small):
    pds = PlotDensitySlices.show_of_data(
        of_data_sample_small, construct_ks_data=True, rotate_molecule_mode="None"
    )
    pds.plot_difference_interactive()
