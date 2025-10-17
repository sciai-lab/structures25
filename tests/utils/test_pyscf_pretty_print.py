import pyscf
import pytest

from mldft.utils.pyscf_pretty_print import (
    mol_to_str,
    mole_to_sum_formula,
    monkey_patch_pyscf_mol_str,
)

parametrize_with_mol_and_sum_formulas = pytest.mark.parametrize(
    ["mol", "sum_formula", "sum_formula_subscript"],
    [
        [pyscf.gto.M("C 0 0 0; O 0 0 1", spin=None), "CO", "CO"],
        [pyscf.gto.M("H 0 0 1; O 0 0 2; H 0 0 0", spin=None), "H2O", "H₂O"],
        [pyscf.gto.M("C 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3; H 0 0 4", spin=None), "CH4", "CH₄"],
    ],
)


@parametrize_with_mol_and_sum_formulas
def test_mole_to_sum_formula(
    mol: pyscf.gto.Mole, sum_formula: str, sum_formula_subscript: str
) -> None:
    """Test the mole_to_sum_formula function."""
    assert mole_to_sum_formula(mol) == sum_formula
    assert mole_to_sum_formula(mol, use_subscript=True) == sum_formula_subscript


@parametrize_with_mol_and_sum_formulas
def test_mol_to_str(mol: pyscf.gto.Mole, sum_formula: str, sum_formula_subscript: str) -> None:
    """Test the mol_to_str function."""
    assert mol_to_str(mol) == f"<pyscf.gto.mole.Mole with sum formula ({sum_formula})>"


@parametrize_with_mol_and_sum_formulas
def test_monkey_patch_pyscf_mol_str(
    mol: pyscf.gto.Mole, sum_formula: str, sum_formula_subscript: str
) -> None:
    """Test the monkey_patch_pyscf_mol_str function."""
    default_str = pyscf.gto.Mole.__str__
    monkey_patch_pyscf_mol_str()
    assert mol.__str__() == f"<pyscf.gto.mole.Mole with sum formula ({sum_formula})>"
    # Reset the monkey-patch
    pyscf.gto.Mole.__str__ = default_str
