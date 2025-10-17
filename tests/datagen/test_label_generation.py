import numpy as np
import pytest

from mldft.datagen.methods.label_generation import calculate_labels
from mldft.utils.molecules import build_mol_with_even_tempered_basis

ATOL = 1e-02


def test_label_generation(ksdft_results_small):
    """Tests if the calculate labels function works as indented and if its fitted energies are
    close to the ones originally calculated."""
    results, initialization, data_of_iteration, mol_orbital_basis = ksdft_results_small
    mol_density_basis = build_mol_with_even_tempered_basis(mol_orbital_basis)

    data = calculate_labels(
        results, initialization, data_of_iteration, mol_orbital_basis, mol_density_basis
    )

    assert pytest.approx(data["ks_e_xc"], abs=ATOL) == data["of_e_xc"]
    # We don't save this anymore
    # assert np.allclose(data["ks_hartree_energy"], data["of_hartree_energy"], atol=3e-02)
    assert pytest.approx(data["ks_e_ext"], abs=ATOL) == data["of_e_ext"]
    assert pytest.approx(data["ks_e_electron"], abs=ATOL) == data["of_e_electron"]
    assert pytest.approx(data["ks_e_kin"], abs=ATOL) == data["of_e_kin"]

    assert data["ks_e_nuc_nuc"] is not None
    assert not np.any(np.isnan(data["of_grad_kin"]))
    assert not np.any(np.isnan(data["of_grad_xc"]))
    assert not np.any(np.isnan(data["of_grad_kinapbe"]))
