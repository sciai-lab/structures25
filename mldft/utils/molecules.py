"""Utilities for working with molecules.

Support for xyz, sdf, chk files and numpy arrays.
"""

import os
from os.path import join
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pyscf
import scipy
import zarr
from pyscf import gto
from pyscf.df import aug_etb
from pyscf.lib.chkfile import load
from rdkit import Chem

# To avoid circular imports and make flake8 happy
if TYPE_CHECKING:
    from mldft.ml.data.components.of_data import OFData


element_symbols_to_numbers = pyscf.data.elements.ELEMENTS_PROTON
numbers_to_element_symbols = pyscf.data.elements.ELEMENTS


def chem_formula_from_atomic_numbers(atomic_numbers: np.ndarray):
    """Returns the chemical formular of a molecule, given the atomic numbers.

    for example: "H_6 C_2 O_1 " for Ethanol
    """
    mol_name = ""
    nums, counts = np.unique(atomic_numbers, return_counts=True)
    for num, count in zip(nums, counts):
        mol_name += f"{numbers_to_element_symbols[num]}_{count} "
    return mol_name


def read_xyz_file(file_path):
    """Reads an xyz file (containing a single molecule) and returns the atomic numbers and
    positions.

    Args:
        file_path: Path to the .xyz file.

    Returns:
        np.ndarray: Array of atomic numbers (A).
        np.ndarray: Array of atomic positions (A, 3).
    """
    try:
        with open(file_path) as f:
            # Read the number of atoms
            num_atoms = int(f.readline().strip())

            # Skip the comment line
            _ = f.readline().strip()

            # Initialize lists to store coordinates and atomic numbers
            coordinates = []
            atomic_numbers = []

            # Read atom data
            for _ in range(num_atoms):
                line = f.readline().split()
                symbol, x, y, z = (
                    line[0],
                    float(line[1]),
                    float(line[2]),
                    float(line[3]),
                )

                # Assuming atomic numbers based on element symbols
                if symbol.isdigit():
                    atomic_number = int(symbol)
                else:
                    # Map element symbols to atomic numbers (customize as needed)

                    atomic_number = element_symbols_to_numbers.get(symbol, 0)

                # Append coordinates and atomic number
                coordinates.append([x, y, z])
                atomic_numbers.append(atomic_number)

        return np.array(atomic_numbers), np.array(coordinates)

    except Exception as e:
        print(f"Error reading .xyz file: {e}")
        return None, None


def build_molecule_np(
    charges: np.ndarray,
    positions: np.ndarray,
    basis: str | dict | None = "6-31G(2df,p)",
    unit: str = "Bohr",
    output: Optional[str] = None,
    spin: Optional[int] = None,
) -> gto.Mole:
    """Given the charges and positions in arrays, build a molecule object using pyscf.

    Args:
        charges: Array of atomic numbers (A).
        positions: Array of atomic positions (A, 3).
        basis: Basis set to use for the molecule.
        unit: Unit of the positions.
        output: Optional output file name.
        spin: Optional spin of the molecule. Defaults to None.
    Returns:
        gto.Mole: A pyscf molecule object.
    """
    # Reformat charges and positions to [(charge, (x, y, z)), ...] for pyscf.
    charge_positions_list = list(zip(charges, positions))
    molecule = gto.M(atom=charge_positions_list, unit=unit, basis=basis, output=output, spin=spin)
    return molecule


def build_mol_with_even_tempered_basis(
    mol: gto.Mole,
    beta: float = 2.5,
    spin: int | None = 0,
) -> gto.Mole:
    """Builds the even-tempered basis set for the given molecule.

    Builds the even-tempered basis set with a factor of :math:`\\beta`
    (defaults to 2.5) in the exponent to approximate
    products of the basis functions given by the mole object. This is normally
    used to build an auxiliary basis to approximate the four-indices Coulomb
    tensor with a two-indices tensor.
    Dimensionalities agree with those reported in [M-OFDFT]_, but we have no way of comparing the exact coefficients.

    .. testsetup::

        from pyscf import gto
        from mldft.ofdft.even_tempered_basis import build_even_tempered_basis # ignore
        from mldft.ofdft.even_tempered_basis import print_basis # ignore

    .. testcode::

        for atm in ["H", "C", "N", "O", "F"]:
            mol = gto.M(atom=f"{atm} 0 0 0; {atm} 0 0 0", basis="6-31G(2df,p)")
            mol_etb = build_even_tempered_basis(mol)
            print_basis(mol_etb)

    .. testoutput::

        H: 6s 3p 1d
        C: 11s 8p 7d 3f 2g
        N: 11s 8p 7d 4f 2g
        O: 11s 8p 7d 4f 2g
        F: 11s 8p 7d 4f 2g

    Args:
        mol: The molecule for which the basis set should be built. It contains
            information about the geometry and the atomic basis set.
        beta: The exponent factor :math:`\\beta` of the even-tempered basis set.
            Should be larger than 1, defaults to 2.5.
        spin: The spin of the molecule. Defaults to 0.

    Returns:
        The molecule with the new even-tempered basis set.
    """
    basis_dict = aug_etb(mol, beta=beta)

    # mol._atom is always using Bohr as a unit. mol.atom on the other hand can
    # use either Bohr or Angstrom. We use mol._atom  and specify the unit
    # as Bohr to avoid any confusion.
    mol_etb = gto.Mole(atom=mol._atom, spin=spin, unit="Bohr")

    mol_etb.basis = basis_dict
    return mol_etb.build()


# string around BasisInfo is needed to avoid circular imports since this function depends on BasisInfo but
# BasisInfo depends on the even tempered basis function in this file
def build_molecule_ofdata(
    ofdata: "OFData", basis: dict | str = None, spin: Optional[int] = None
) -> gto.Mole:
    """Convert the data to a pyscf Mole object.

    Args:
        ofdata: The data to convert.
        basis: The basis set to use for the molecule.
        spin: The spin of the molecule. Defaults to None.

    Returns:
        gto.Mole: A pyscf molecule object.

    Note:
        The basis set can not be inferred from an OFData object alone. If a specific one is
        required a :class:`~mldft.ml.data.components.basis_info.BasisInfo` object should be given or the basis should
        later be set manually.
    """
    charges = (
        ofdata.atomic_numbers
        if isinstance(ofdata.atomic_numbers, np.ndarray)
        else ofdata.atomic_numbers.numpy(force=True)
    )
    positions = ofdata.pos if isinstance(ofdata.pos, np.ndarray) else ofdata.pos.numpy(force=True)
    return build_molecule_np(
        charges=charges,
        positions=positions,
        basis=basis,
        unit="Bohr",
        spin=spin,
    )


def load_scf(chk_file: Path) -> Tuple[dict, dict, list[dict]]:
    """Load the data of a specific iteration from the chk file.

    Args:
        chk_file: Path to the chk file.

    Returns:
        dict: The pyscf object containing the final result of the calculation.
        dict: The density matrix and energy of the first iteration.
        list: A list of dictionaries containing the data of each iteration.
    """
    results = load(chk_file, "Results")
    initialization = load(chk_file, "Initialization")
    max_cycle = results["max_cycle"]
    data_of_all_hf_iterations = []
    for i in range(max_cycle):
        scf_file = load(chk_file, f"KS-iteration/{i:d}")
        if scf_file is None:
            break
        data_of_all_hf_iterations.append(scf_file)
    return results, initialization, data_of_all_hf_iterations


def load_charges_and_positions_sdf(
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Given the path to a .sdf file, build return charges and positions in arrays.

    Args:
        path: Path to the .sdf file.

    Returns:
        np.ndarray: Array of atomic numbers (A).
        np.ndarray: Array of atomic positions (A, 3).

    Raises:
        AssertionError: If the .sdf file contains more than one molecule.
    """
    with Chem.SDMolSupplier(path.as_posix(), removeHs=False) as supply:
        mol = next(supply)
        assert len(list(supply)) == 1, "Only one molecule per file is supported."
        # Get the 3D coordinates (positions) for each atom and save
        conformer = mol.GetConformer()
        num_atoms = mol.GetNumAtoms()
        atomic_num_array = np.zeros(num_atoms, dtype=np.uint8)
        position_array = np.zeros((num_atoms, 3))
        for atom_idx in range(num_atoms):
            atom = mol.GetAtomWithIdx(atom_idx)
            position = conformer.GetAtomPosition(atom_idx)
            atomic_num = atom.GetAtomicNum()
            atomic_num_array[atom_idx] = atomic_num
            position_array[atom_idx] = [position.x, position.y, position.z]
    return atomic_num_array, position_array


def print_basis(mol: gto.Mole):
    """Prints the number of s, p, d, f, g orbitals for each atom in the molecule.

    Args:
        mol: The molecule for which the summary should be printed.

    Returns:
        None
    """
    angular_momentum = ["s", "p", "d", "f", "g", "h"]

    if type(mol.basis) is str:
        atom_types = [numbers_to_element_symbols[num] for num in np.unique(mol.atom_charges())]
        atomic_basis = {
            atom_type: gto.basis.load(mol.basis, atom_type) for atom_type in atom_types
        }
    else:
        atomic_basis = mol.basis

    for atomic_number, aos in atomic_basis.items():
        print(atomic_number, end=": ")
        atom_ls = np.zeros(len(angular_momentum), dtype=np.uint32)

        for ao in aos:
            atom_ls[int(ao[0])] += 1

        print(" ".join(f"{n}{angular_momentum[ang]}" for ang, n in enumerate(atom_ls) if n > 0))


def construct_aux_mol(
    mol: gto.Mole, aux_basis_name: str = "even_tempered_2.5", unit="Bohr", spin=None
) -> gto.Mole:
    """Constructs an auxiliary molecule in the basis specified by aux_basis_name.

    Possible options for aux_basis_name are:
    f"even_tempered_{beta}"
    f"even_tempered_{beta}_from_{basis_name}"
    f"{basis_name}"
    Args:
        mol: The molecule for which the auxiliary molecule should be constructed.
        aux_basis_name: The name of the auxiliary basis to construct.
        unit: The unit of the coordinates.
        spin: The spin of the molecule. Defaults to None.
    Note:
        The second option f"even_tempered_{beta}_from_{basis_name}" effects the minimal and maximal exponents of the basis set,
        the density of the basis function between them doesn't change.
    """
    if type(aux_basis_name) is not str:
        raise TypeError(f"aux_basis_name must be a string not {type(aux_basis_name)}")
    if aux_basis_name.startswith("even_tempered"):
        temp_mol = mol
        split_name = aux_basis_name.split("_")

        if len(split_name) > 3:
            if split_name[3] == "from":
                temp_mol = build_molecule_np(
                    mol.atom_charges(),
                    mol.atom_coords(),
                    split_name[4],
                    unit=unit,
                    spin=spin,
                )
        aux_mol = build_mol_with_even_tempered_basis(
            temp_mol, beta=float(split_name[2]), spin=spin
        )
    elif aux_basis_name[:2] == "20":
        """zarr_path = join( os.environ["DFT_DATA"], "Basis_sets", "logs", aux_basis_name,

        "basis_set.zarr.zip", )
        """
        zarr_path = join(
            os.environ["DFT_MODELS"],
            "basis_set_fitting",
            "logs",
            aux_basis_name,
            "basis_set.zarr.zip",
        )
        if not os.path.exists(zarr_path):
            raise FileNotFoundError(f" The path to the basis set :{zarr_path} was not found.")
        atomic_numbers = np.unique(mol.atom_charges())
        (
            angmon_dict,
            coeffs_dict,
            exps_dict,
            exps_index_dict,
            coeffs_index_dict,
        ) = load_basis_from_zarr(zarr_path, atomic_numbers)
        pyscf_dict = dict_to_pyscf_dict(
            angmon_dict, coeffs_dict, exps_dict, exps_index_dict, coeffs_index_dict
        )
        charge_positions_list = list(zip(mol.atom_charges(), mol.atom_coords()))
        aux_mol = gto.M(
            atom=charge_positions_list,
            unit="B",
            basis=pyscf_dict,
            output=mol.output,
            spin=mol.spin,
        )
        aux_mol.build()
    else:
        aux_mol = build_molecule_np(
            mol.atom_charges(), mol.atom_coords(), aux_basis_name, unit=unit, spin=spin
        )
    return aux_mol


def load_basis_from_zarr(zarr_path: Path, atomic_numbers) -> Dict[str, np.ndarray]:
    """Load the basis set from a zarr file.

    The basis set is represented using 5 different dictionaries,
    containing numpy arrays for each different atom type.
    Args:
        zarr_path: Path to the zarr file.
        atomic_numbers: The atomic numbers of the atoms in the molecule.
    Returns:
        angmon_dict: A dictionary containing a 1d array of the different angular momentum of the basis functions each atom type contains.
        coeffs_dict: A dictionary containing a flatted 1d array of the coefficients of the basis functions each atom type contains.
        exps_dict: A dictionary containing a flatted 1d array of the exponents of the basis functions each atom type contains.
        exps_index_dict: A dictionary containing a 2d array providing information about how the exponents of exps_dict are grouped into different shells.
        coeffs_index_dict: A dictionary containing a 2d array providing information about how the coefficients of coeffs_dict are grouped into different shells.
    """
    angmon_dict = {}
    coeffs_dict = {}
    exps_dict = {}
    exps_index_dict = {}
    coeffs_index_dict = {}
    elements_list = [numbers_to_element_symbols[atomic_number] for atomic_number in atomic_numbers]
    if not os.path.isfile(zarr_path):
        raise FileNotFoundError(f"There was no basis set found at {zarr_path}")
    with zarr.open(zarr_path, mode="r") as root:
        for element in elements_list:
            if element + "_angmon" in root:
                angmon_dict[element] = np.asarray(root[element + "_angmon"], dtype=np.int32)
                coeffs_dict[element] = np.asarray(root[element + "_coeffs"], dtype=np.float64)
                exps_dict[element] = np.asarray(root[element + "_exponents"], dtype=np.float64)
            else:
                raise ValueError(f"{element}_angmon is not in the zarr file.")
            if element + "_exponents_index" in root:
                exps_index_dict[element] = np.asarray(
                    root[element + "_exponents_index"], dtype=np.int32
                )
            else:
                exps_index_dict[element] = np.ones(exps_dict[element].shape[0], dtype=np.int32)
            if element + "_coeffs_index" in root:
                coeffs_index_dict[element] = np.asarray(
                    root[element + "_coeffs_index"], dtype=np.int32
                )
            else:
                coeffs_index_dict[element] = None

    return angmon_dict, coeffs_dict, exps_dict, exps_index_dict, coeffs_index_dict


def dict_to_pyscf_dict(
    angmon_dict: Dict[str, np.ndarray],
    coeffs_dict: Dict[str, np.ndarray],
    exps_dict: Dict[str, np.ndarray],
    exps_index_dict: Dict[str, np.ndarray],
    coeffs_index_dict: Dict[str, np.ndarray],
) -> Dict[str, List[Tuple[int, Tuple[float, float]]]]:
    """Translates the basis set from a dictionary format used to store then to a format that can be
    used by pyscf.

    Args:
        angmon_dict: A dictionary containing a 1d array of the different angular momentum of the basis functions each atom type contains.
        coeffs_dict: A dictionary containing a flatted 1d array of the coefficients of the basis functions each atom type contains.
        exps_dict: A dictionary containing a flatted 1d array of the exponents of the basis functions each atom type contains.
        exps_index_dict: A dictionary containing a 2d array providing information about how the exponents of exps_dict are grouped into different shells.
        coeffs_index_dict: A dictionary containing a 2d array providing information about how the coefficients of coeffs_dict are grouped into different shells.
    Returns:
        pyscf_dict: A dictionary containing the basis set in a format that can be used by
    """
    if coeffs_index_dict is not None and not all(
        [coeffs_index is None for coeffs_index in coeffs_index_dict.values()]
    ):
        for key in coeffs_index_dict.keys():
            grouped_data = {}
            for item in coeffs_index_dict[key]:
                if item[1] in grouped_data:
                    grouped_data[item[1]].append(item)
                else:
                    grouped_data[item[1]] = [item]
            coeffs_index_dict[key] = grouped_data.values()
        constructed_coeffs_dict = construct_coeffs(coeffs_dict, coeffs_index_dict)
    else:
        constructed_coeffs_dict = coeffs_dict
    pyscf_dict = {}
    for key in exps_dict.keys():
        list = []
        exp_list = np.split(exps_dict[key].flatten(), exps_index_dict[key].cumsum())
        for angmon, coeffs, exps in zip(angmon_dict[key], constructed_coeffs_dict[key], exp_list):
            for coeff_col in coeffs.T:
                non_zero_mask = coeff_col != 0.0
                list.append(
                    (
                        angmon.item(),
                        *[
                            (exp.item(), coeff.item())
                            for exp, coeff in zip(exps[non_zero_mask], coeff_col[non_zero_mask])
                        ],
                    )
                )
            # list.append((angmon.item(), (exp, coeff)))
        pyscf_dict[key] = list
    return pyscf_dict


def construct_coeffs(coeffs_dict: Dict[str, np.ndarray], coeffs_index_dict: Dict[str, np.ndarray]):
    """Reads out coeffs_dict and coeffs_index_dict to construct a list of coeffs for each shell.

    The coeffs are represented in a block diagonal fashion, where each block corresponds to a contrachted basis function.
    This was implemented to simplify the use of differentiable integrals.
    Args:
        coeffs_dict: A dictionary containing a flatted 1d array of the coefficients of the basis functions each atom type contains.
        coeffs_index_dict: A dictionary containing a 2d array providing information about how the coefficients of coeffs_dict are grouped into different shells.
    Returns:
        res_dict: A dictionary containing the coefficients of the basis functions each atom type contains in a block diagonal format.
    """
    res_dict = {}
    for key in coeffs_dict.keys():
        coeffs_blocks = []
        for block in coeffs_index_dict[key]:
            block_diag_list = []
            for ones, _, start_index, coeffs_index, exponent_index in block:
                if ones > 0:
                    block_diag_list.append(np.eye(ones, dtype=np.float64))
                if start_index is not None:
                    block_diag_list.append(
                        coeffs_dict[key][
                            start_index : start_index + coeffs_index * exponent_index
                        ].reshape(coeffs_index, exponent_index)
                    )
            coeffs_blocks.append(scipy.linalg.block_diag(*block_diag_list))
        res_dict[key] = coeffs_blocks
    return res_dict


def geometry_to_string(mol: gto.Mole, unit: str = "Angstrom"):
    """Convert the geometry of a molecule to a string.

    Args:
        mol: The molecule.
        unit: The unit of the coordinates.

    Returns:
        The geometry as a string.
    """
    return "\n".join(
        f"{mol.atom_symbol(i)} {mol.atom_coord(i, unit=unit)[0]:.8f} {mol.atom_coord(i, unit=unit)[1]:.8f} {mol.atom_coord(i, unit=unit)[2]:.8f}"
        for i in range(mol.natm)
    )


def check_atom_types(mol: gto.Mole, atom_types: np.ndarray) -> None:
    """Check if all atoms in the molecule are of a certain type.

    Args:
        mol: The molecule.
        atom_types: The atom types to check for.

    Raises:
        AssertionError: If an atom in the molecule is not of the specified type.
    """
    atom_charges = np.unique(mol.atom_charges())

    assert all(
        atom in atom_types for atom in atom_charges
    ), f"Allowed atoms {atom_types}, found {atom_charges}"
