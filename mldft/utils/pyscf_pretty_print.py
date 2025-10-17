import numpy as np
import pyscf

SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def mole_to_sum_formula(mol: pyscf.gto.mole.Mole, use_subscript=False) -> str:
    """Converts a pyscf.gto.mole.Mole to a sum formula.

    Args:
        mol: Molecule to convert.
        use_subscript: Whether to use subscript for the number of atoms.

    Returns:
        sum_formula: Sum formula of the molecule.
    """
    atom_charges, counts = np.unique(mol.atom_charges(), return_counts=True)
    elements = [pyscf.data.elements.ELEMENTS[c] for c in atom_charges]
    element_counts = dict(zip(elements, counts))

    # Sort the elements as per Hill system
    if "C" in element_counts:
        sorted_elements = ["C"]
        if "H" in element_counts:
            sorted_elements.append("H")
        sorted_elements += sorted(e for e in elements if e not in ["C", "H"])
    else:
        sorted_elements = sorted(elements)

    sum_formula = "".join(
        f"{el}{element_counts[el]}" if element_counts[el] > 1 else el for el in sorted_elements
    )

    if use_subscript:
        sum_formula = sum_formula.translate(SUB)

    return sum_formula


def mol_to_str(mol: pyscf.gto.mole.Mole) -> str:
    """String representation of a pyscf.gto.mole.Mole, with more information than the default."""
    return f"<pyscf.gto.mole.Mole with sum formula ({mole_to_sum_formula(mol)})>"


def monkey_patch_pyscf_mol_str() -> None:
    """Monkey-patch pyscf.gto.Mole.__str__ to produce a more useful string representation.

    To use, call this function once at the beginning of your script: ``` from
    mldft.utils.pyscf_pretty_print import monkey_patch_pyscf_mol_str monkey_patch_pyscf_mol_str()
    ```
    """
    pyscf.gto.Mole.__str__ = mol_to_str
