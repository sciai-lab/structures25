from pyscf import gto

from mldft.utils import molecules


def test_build_mol_with_even_tempered_basis(molecule_small: gto.Mole):
    """Tests if the build_mol_with_even_tempered_basis function works as intended."""
    mol = molecules.build_mol_with_even_tempered_basis(molecule_small)

    assert mol._atom == molecule_small._atom
    assert mol.spin == molecule_small.spin
    assert mol.nelectron == molecule_small.nelectron
