from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from pyscf import gto

from mldft.datagen.methods.ksdft_calculation import ksdft
from mldft.utils.molecules import load_scf


@pytest.fixture(params=("CDIIS",))
def diis_method(request):
    """Fixture for the diis method.

    For now, only CDIIS is supported and hence tested.
    """
    return request.param


def test_ksdft(molecule_small: gto.Mole, diis_method: str):
    """Test the ksdft and the load_scf function."""
    # mol = build_molecule_np(charges, positions)
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        filesave_path = temp_dir_path / "test.chk"
        assert not filesave_path.exists()
        ksdft(molecule_small, filesave_path, diis_method=diis_method)
        assert filesave_path.exists()
        results, initialization, data_of_ks_iteration = load_scf(filesave_path)
        assert "converged" in results
        assert "total_energy" in results
        assert "molecular_coeffs_orbitals" in results
        assert "occupation_numbers_orbitals" in results
        assert "max_cycle" in results
        assert "name_xc_functional" in results
        assert "init_guess" in results
        assert "convergence_tolerance" in results
        assert "diis_start_cycle" in results
        assert "diis_space" in results
        assert "first_density_matrix" in initialization
        assert "first_total_energy" in initialization
        assert (
            initialization["first_density_matrix"].shape[0]
            == results["molecular_coeffs_orbitals"].shape[0]
        )

        for i, data in enumerate(data_of_ks_iteration):
            assert type(data) == dict
            assert "diis_coefficients" in data
            assert "occupation_numbers_orbitals" in data
            assert "molecular_coeffs_orbitals" in data
            assert "total_energy" in data
            assert "coulomb_energy" in data
            assert "exchange_correlation_energy" in data
            assert data["diis_coefficients"].shape <= (results["diis_space"] + 1,)
            assert (
                data["occupation_numbers_orbitals"].shape
                == results["occupation_numbers_orbitals"].shape
            )
            assert (
                data["molecular_coeffs_orbitals"].shape
                == results["molecular_coeffs_orbitals"].shape
            )
