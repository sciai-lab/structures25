import pytest
import torch
from pyscf import gto

from mldft.ml.data.components.convert_transforms import ToTorch
from mldft.ml.data.components.of_data import OFData
from mldft.ofdft import basis_integrals, initial_guess
from mldft.utils.molecules import build_mol_with_even_tempered_basis


def test_hueckel_guess(molecule_medium: gto.Mole):
    """Tests shape and electron number of the hueckel_guess."""
    mol = build_mol_with_even_tempered_basis(molecule_medium)
    sample = OFData.minimal_sample_from_mol(mol)
    sample = ToTorch(float_dtype=torch.float64)(sample)
    guess = initial_guess.hueckel_guess(
        sample, mol, ks_basis=molecule_medium.basis, transform=False
    )

    assert guess.shape[0] == mol.nao
    assert guess.ndim == 1
    normalization_vector = basis_integrals.get_normalization_vector(mol)
    n_guess = guess.numpy() @ normalization_vector
    assert molecule_medium.nelectron == pytest.approx(n_guess)


def test_minao_guess(molecule_medium: gto.Mole):
    """Tests shape and electron number of the minao_guess."""
    mol = build_mol_with_even_tempered_basis(molecule_medium)
    sample = OFData.minimal_sample_from_mol(mol)
    sample = ToTorch(float_dtype=torch.float64)(sample)
    guess = initial_guess.minao_guess(sample, mol, ks_basis=molecule_medium.basis, transform=False)

    assert guess.shape[0] == mol.nao
    assert guess.ndim == 1
    normalization_vector = basis_integrals.get_normalization_vector(mol)
    n_guess = guess.numpy() @ normalization_vector
    assert molecule_medium.nelectron == pytest.approx(n_guess)


def test_dumb_guess(molecule_medium: gto.Mole):
    """Tests the dumb_guess function."""
    mol = build_mol_with_even_tempered_basis(molecule_medium)
    normalization_vector = basis_integrals.get_normalization_vector(mol)
    sample = OFData.minimal_sample_from_mol(mol)
    sample = ToTorch(float_dtype=torch.float64)(sample)
    guess = initial_guess.dumb_guess(sample, mol)

    assert guess.shape[0] == normalization_vector.shape[0]
    assert torch.all(guess[normalization_vector == 0] == 0)
    n_guess = guess.numpy() @ normalization_vector
    assert molecule_medium.nelectron == pytest.approx(n_guess)
