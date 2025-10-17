from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import zarr

from mldft.datagen.methods.label_generation import calculate_labels
from mldft.datagen.methods.save_labels_in_zarr_file import save_density_fitted_data
from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.of_data import OFData
from mldft.utils.molecules import build_mol_with_even_tempered_basis


def test_save_data_to_zarr(ksdft_results_small):
    """Loads a .chk file and a .npz file and runs density fitting on the data.

    Args:
        ksdft_results_small: the tuple of data from a ksdft calculation

    Returns:
        data: a dict of all data relevant for ofdft calculations and comparisons. (see calculate_labels for details)
    """

    results, initialization, data_of_iteration, mol_with_orbital_basis = ksdft_results_small
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        filesave_path = temp_dir_path / "test_molecule.zarr.zip"
        mol_with_density_basis = build_mol_with_even_tempered_basis(mol_with_orbital_basis)
        data = calculate_labels(
            results,
            initialization,
            data_of_iteration,
            mol_with_orbital_basis,
            mol_with_density_basis,
        )
        ks_string = "This is a string of the ks_basis"
        path_to_basis_info = "This is the relative path to the basis info file"
        save_density_fitted_data(
            filesave_path,
            mol_with_density_basis,
            ks_string,
            path_to_basis_info,
            data,
            "MolId123",
        )
        # Testing that all data is saved where it is supposed to be
        root = zarr.open(filesave_path, mode="r")
        geometry = root["geometry"]
        assert root["of_labels/n_scf_steps"][()] == data["of_coeffs"].shape[0]
        assert geometry["mol_id"][()] == "MolId123"
        assert np.allclose(geometry["atom_pos"], mol_with_density_basis.atom_coords())
        assert np.allclose(geometry["atomic_numbers"], mol_with_density_basis.atom_charges())

        of_labels = root["of_labels"]
        assert of_labels["basis"][()] == path_to_basis_info

        of_energies = of_labels["energies"]
        assert np.allclose(of_energies["e_kin"], data["of_e_kin"])
        assert np.allclose(of_energies["e_xc"], data["of_e_xc"])
        assert np.allclose(of_energies["e_hartree"], data["of_e_hartree"])
        assert np.allclose(of_energies["e_ext"], data["of_e_ext"])

        of_spacial = of_labels["spatial"]
        assert np.allclose(of_spacial["coeffs"], data["of_coeffs"])
        assert np.allclose(of_spacial["grad_kin"], data["of_grad_kin"])
        assert np.allclose(of_spacial["grad_xc"], data["of_grad_xc"])

        ks_labels = root["ks_labels"]
        assert ks_labels["basis"][()] == ks_string

        ks_energies = ks_labels["energies"]
        assert np.allclose(ks_energies["e_kin"], data["ks_e_kin"])
        assert np.allclose(ks_energies["e_xc"], data["ks_e_xc"])
        assert np.allclose(ks_energies["e_ext"], data["ks_e_ext"])
        assert np.allclose(ks_energies["e_nuc_nuc"], data["ks_e_nuc_nuc"])

        # Testing for compatibility with load_sample
        atomic_numbers = np.unique(mol_with_orbital_basis.atom_charges())
        n_scf_iterations = data["of_coeffs"].shape[0]
        basis_info = BasisInfo.from_atomic_numbers_with_even_tempered_basis(
            atomic_numbers, basis=mol_with_orbital_basis.basis, beta=2.5
        )

        for i in range(n_scf_iterations):
            of_data = OFData.from_file(filesave_path, i, basis_info)
            assert np.allclose(of_data.coeffs, data["of_coeffs"][i])
            assert np.allclose(of_data.gradient_label, data["of_grad_kin"][i])
            assert np.allclose(of_data.energy_label, data["of_e_kin"][i])
            assert np.allclose(
                of_data.ground_state_coeffs, data["of_coeffs"][n_scf_iterations - 1]
            )
