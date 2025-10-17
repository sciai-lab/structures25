import pyscf
import rdkit
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


def pyscf_to_rdkit(
    mol: pyscf.gto.mole.Mole, infer_bonds=True, catch_errors=False
) -> rdkit.Chem.rdchem.Mol:
    """Convert a PySCF molecule to an RDKit molecule.

    Args:
        mol: The PySCF molecule.
        infer_bonds: Whether to infer bonds from the geometry.
        catch_errors: Whether to catch sanitization errors.

    Returns:
    """
    xyz_string = mol.tostring("xyz")
    rdkit_mol = Chem.MolFromXYZBlock(xyz_string)

    if infer_bonds:
        rdkit_mol = Chem.Mol(rdkit_mol)
        rdDetermineBonds.DetermineConnectivity(rdkit_mol)

    Chem.SanitizeMol(rdkit_mol, catchErrors=catch_errors)

    return rdkit_mol
