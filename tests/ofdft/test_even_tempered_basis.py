import numpy as np
import pytest

from mldft.utils.molecules import build_mol_with_even_tempered_basis


@pytest.mark.parametrize("beta", [1.2, 1.5, 2.0, 5.0])
def test_build_even_tempered_basis(molecule_small, beta):
    """Test ratio of basis function exponents."""
    mol = molecule_small
    mol_etb = build_mol_with_even_tempered_basis(mol, beta=beta)

    for atomic_numbers, aos in mol_etb.basis.items():
        for ao1, ao2 in zip(aos[:-1], aos[1:]):
            if ao1[0] == ao2[0]:  # same angular momentum
                assert np.isclose(ao1[1][0], beta * ao2[1][0])
