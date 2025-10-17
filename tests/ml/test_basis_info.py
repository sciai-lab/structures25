import basis_set_exchange
import numpy as np
import pyscf
import pytest
from e3nn.o3 import Irreps
from pyscf import gto

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ofdft.basis_integrals import get_overlap_matrix
from mldft.utils.molecules import build_mol_with_even_tempered_basis


@pytest.mark.parametrize(
    "basis_name, elements, irreps_per_atom, seed",
    [
        [
            "cc-pvdz",
            ["H", "C", "O"],
            ["2x0e+1x1o", "3x0e+2x1o+1x2e", "3x0e+2x1o+1x2e"],
            0,
        ],
        [
            "cc-pvdz",
            ["C", "H", "O"],
            ["3x0e+2x1o+1x2e", "2x0e+1x1o", "3x0e+2x1o+1x2e"],
            1,
        ],
        [
            "6-31G(2df,p)",
            ["H", "C", "O"],
            ["2x0e+2x1o", "3x0e+2x1o+2x2e+1x3o", "3x0e+2x1o+2x2e+1x3o"],
            2,
        ],
        ["sto-3g", ["H", "C", "O"], ["1x0e", "2x0e+1x1o", "2x0e+1x1o"], 3],
    ],
)
def test_from_nwchem(basis_name, elements, irreps_per_atom, seed):
    """Test the BasisInfo.from_nwchem method using basis sets from the Basis Set Exchange."""
    np.random.seed(seed)

    nwchem_string = basis_set_exchange.api.get_basis(basis_name, elements=elements, fmt="nwchem")
    atomic_numbers = np.array([pyscf.gto.param.ELEMENTS_PROTON[symbol] for symbol in elements])

    basis_info = BasisInfo.from_nwchem(nwchem_string)
    assert np.allclose(basis_info.atomic_numbers, np.sort(atomic_numbers))

    basis_info = BasisInfo.from_nwchem(nwchem_string, atomic_numbers=atomic_numbers)
    assert np.allclose(basis_info.atomic_numbers, atomic_numbers)
    basis_dim_per_atom = np.array([Irreps(irreps).dim for irreps in irreps_per_atom])
    assert np.allclose(basis_info.basis_dim_per_atom, basis_dim_per_atom)
    assert np.all(
        np.equal([str(irreps) for irreps in basis_info.irreps_per_atom], irreps_per_atom)
    )

    # use random subset of the elements
    requested_mask = np.random.randint(0, 2, len(elements), dtype=bool)
    if not np.any(requested_mask):
        requested_mask[0] = True

    basis_info = BasisInfo.from_nwchem(
        nwchem_string, atomic_numbers=atomic_numbers[requested_mask]
    )
    assert np.allclose(basis_info.atomic_numbers, atomic_numbers[requested_mask])
    assert np.allclose(basis_info.basis_dim_per_atom, basis_dim_per_atom[requested_mask])
    assert np.all(
        np.equal(
            [str(irreps) for irreps in basis_info.irreps_per_atom],
            np.fromiter(irreps_per_atom, dtype=object)[requested_mask],
        )
    )


def test_from_atomic_numbers(molecule_all):
    """Just test that it runs through and that the atomic numbers and basis are correct."""
    atomic_numbers = np.unique(molecule_all.atom_charges())
    basis_info = BasisInfo.from_atomic_numbers_with_even_tempered_basis(
        atomic_numbers, basis=molecule_all.basis
    )
    basis_dict = basis_info.basis_dict
    mol_new_basis = gto.M(atom=molecule_all.atom, basis=basis_dict)
    mol_even_tempered_basis = build_mol_with_even_tempered_basis(molecule_all)
    assert np.all(basis_info.atomic_numbers == atomic_numbers)
    assert mol_new_basis.basis == mol_even_tempered_basis.basis


def test_one_basis_info_for_all_mols(molecule_all):
    """Test that one basis dict with all atoms from
    BasisInfo.from_atomic_numbers_with_even_tempered_basis method works for all molecules."""
    atomic_numbers = np.arange(1, 19)
    basis_info = BasisInfo.from_atomic_numbers_with_even_tempered_basis(
        atomic_numbers, basis=molecule_all.basis
    )
    basis_dict = basis_info.basis_dict
    mol_new_basis = gto.M(atom=molecule_all._atom, basis=basis_dict, unit="Bohr")
    mol_new_basis.decontract_basis()
    mol_even_tempered_basis = build_mol_with_even_tempered_basis(molecule_all)
    # Test whether the overlap matrix is the same for the new molecule with the basis set build from BasisInfo
    overlap_new_basis = get_overlap_matrix(mol_new_basis)
    overlap_even_tempered_basis = get_overlap_matrix(mol_even_tempered_basis)
    assert np.allclose(overlap_new_basis, overlap_even_tempered_basis)


def test_from_mol(molecule_all):
    mol_even_tempered_basis = build_mol_with_even_tempered_basis(molecule_all, beta=2.5)

    atomic_numbers = np.unique(molecule_all.atom_charges())

    basis_info_from_mol = BasisInfo.from_mol(mol_even_tempered_basis)
    basis_info_from_atomic_numbers = BasisInfo.from_atomic_numbers_with_even_tempered_basis(
        atomic_numbers, basis=molecule_all.basis, beta=2.5
    )

    assert basis_info_from_mol == basis_info_from_atomic_numbers


def test___eq__(molecule_all):
    atomic_numbers = np.unique(molecule_all.atom_charges())
    basis_info_from_atomic_numbers_1 = BasisInfo.from_atomic_numbers_with_even_tempered_basis(
        atomic_numbers, basis=molecule_all.basis, beta=2.5
    )
    basis_info_from_atomic_numbers_2 = BasisInfo.from_atomic_numbers_with_even_tempered_basis(
        atomic_numbers, basis=molecule_all.basis, beta=2.5
    )
    basis_info_from_atomic_numbers_3 = BasisInfo.from_atomic_numbers_with_even_tempered_basis(
        atomic_numbers, basis=molecule_all.basis, beta=1.2
    )

    assert basis_info_from_atomic_numbers_1 == basis_info_from_atomic_numbers_2
    assert not basis_info_from_atomic_numbers_1 == basis_info_from_atomic_numbers_3


def test_is_subset(molecule_all):
    atomic_numbers = np.unique(molecule_all.atom_charges())
    atomic_numbers_other = np.asarray([12], dtype=np.int8)
    basis_info_from_atomic_numbers_full = BasisInfo.from_atomic_numbers_with_even_tempered_basis(
        atomic_numbers, basis=molecule_all.basis, beta=2.5
    )
    basis_info_from_atomic_numbers_subset = BasisInfo.from_atomic_numbers_with_even_tempered_basis(
        np.asarray([atomic_numbers[0]], dtype=np.int8),
        basis=molecule_all.basis,
        beta=2.5,
    )
    basis_info_from_atomic_numbers_other = BasisInfo.from_atomic_numbers_with_even_tempered_basis(
        atomic_numbers_other, basis=molecule_all.basis, beta=2.5
    )

    assert basis_info_from_atomic_numbers_subset.is_subset(basis_info_from_atomic_numbers_full)
    assert not basis_info_from_atomic_numbers_other.is_subset(basis_info_from_atomic_numbers_full)
