"""The :class:`BasisInfo` class holds ML-relevant information about the basis."""

import os

import numpy as np
import pyscf
import torch
from e3nn.o3 import Irreps
from loguru import logger
from omegaconf import OmegaConf

from mldft.ofdft.basis_integrals import get_normalization_vector
from mldft.utils.molecules import build_mol_with_even_tempered_basis, construct_aux_mol


def _irreps_list_to_array(irreps_list: list[Irreps | str]) -> np.ndarray[Irreps]:
    """Convert a list of irreps, as :class:`str` or :class:`~e3nn.o3.Irreps`, to a 1D array of
    :class:`~e3nn.o3.Irreps`.

    Args:
        irreps_list: List of irreps, as :class:`str` or :class:`~e3nn.o3.Irreps`.

    Returns:
        irreps_array: 1D array of :class:`~e3nn.o3.Irreps`.
    """
    #  use np.fromiter to prevent irreps of equal shape being packed into a multidimensional array
    #  (as would happen with np.array)
    return np.fromiter([Irreps(irreps) for irreps in irreps_list], dtype=object)


class BasisInfo:
    """Information about an OF basis for the LCAB Ansatz (see :mod:`~mldft.ofdft), to be used in
    data loading. It does not directly include the actual shape of the basis functions, i.e.
    exponents, contraction coefficients etc., but the attribute :attr:`basis_dict` can be passed to
    :class:`pyscf.gto.Mole` as the ``basis`` keyword to construct a pyscf molecule.

    Attributes:
        atomic_numbers: Atomic numbers of the atom types in the basis. Shape (n_atom_types,).
        basis_dict: Dict of basis sets for each atom type, compatible with pyscf. Length n_atom_types.
        basis_dim_per_atom: Number of basis functions per atom type. Shape (n_atom_types,).
        irreps_per_atom: Irreps of the basis functions per atom type. Shape (n_atom_types,).
        atomic_number_to_atom_index: Mapping from atomic number to atom type index in the basis.
            Shape (max(atomic_numbers)+1,).
            e.g. for H, C, O: [-1, 0, -1, -1, -1, -1, 1, -1, 2]
        integrals: Integrals over the basis functions. Array of arrays of length n_atom_types.
        atom_ind_to_basis_function_ind: Mapping list from the atomic number index in the basis to
            index basis function indices of a concatenated basis function array.
            e.g. for H, C, O and basis dimensions 4, 5, 9:
            [[0,1,2,3], [4,5,6,7,8], [9,10,11,12,13,14,15,16,17]]
        l_per_basis_func: Orbital number l for each basis function. Shape (n_basis,).
        m_per_basis_func: Orbital number m for each basis function. Shape (n_basis,).
        shell_beginning_mask: Mask with ones at the beginning of each shell. Shape (n_basis,).
        basis_func_to_shell: Mapping from basis function index to shell index. Shape (n_basis,).
        shell_to_first_basis_func: Mapping from shell index to beginning of basis functions in that shell.
            Shape (n_shells,).
        n_shells_per_atom: Number of shells per atom type. Shape (n_atom_types,).
        l_per_shell: Orbital number l for each shell. Shape (n_shells,).
        m_per_shell: Orbital number m for each shell. Shape (n_shells,).


    .. warning::
        This is most likely only correct after transforming into the e3nn convention
        (and otherwise wrong for l=1).
    """

    def __init__(
        self,
        atomic_numbers: np.ndarray[np.uint8],
        basis_dict: dict[str, list | str],
        irreps_per_atom: np.ndarray[Irreps] | list[Irreps | str],
        integrals: np.ndarray[np.ndarray] | list[np.ndarray],
    ) -> None:
        """Initialize the BasisInfo object.

        Args:
            atomic_numbers: Atomic numbers of the atoms in the basis. Shape (n_atom_types,).
            basis_dict: Dict of basis sets for each atom. Length {n_atom_types}.
            irreps_per_atom: Irreps of the basis functions per atom. Shape (n_atom_types,).
            integrals: Integrals over the basis functions. List of (n_atom_types,).
        """
        self.atomic_numbers = atomic_numbers
        self.basis_dict = basis_dict
        self.irreps_per_atom = _irreps_list_to_array(irreps_per_atom)
        self.basis_dim_per_atom = np.array([irreps.dim for irreps in self.irreps_per_atom])
        self.integrals = np.fromiter(
            [integrals_for_atom for integrals_for_atom in integrals], dtype=object
        )

        # construct the mapping from atomic number to atom index in the basis
        self.atomic_number_to_atom_index = -np.ones(
            np.max(self.atomic_numbers) + 1, dtype=np.int32
        )
        self.atomic_number_to_atom_index[self.atomic_numbers] = np.arange(len(self.atomic_numbers))
        self.atom_ind_to_basis_function_ind = np.split(
            np.arange(sum(self.basis_dim_per_atom)),
            np.cumsum(self.basis_dim_per_atom)[:-1],
        )

        basis_func_to_l_per_atom_type = []
        basis_func_to_m_per_atom_type = []
        n_shells_per_atom = []
        for irrep in self.irreps_per_atom:
            assert np.all(irrep.ls[:-1] <= irrep.ls[1:]), f"ls are not sorted: {irrep.ls}"
            ls, count = np.unique(irrep.ls, return_counts=True)
            basis_func_to_l_per_atom_type.append(
                np.concatenate(
                    [np.full(count * (2 * l_no + 1), l_no) for l_no, count in zip(ls, count)]
                )
            )
            basis_func_to_m_per_atom_type.append(
                np.concatenate(
                    [np.tile(np.arange(-l_no, l_no + 1), count) for l_no, count in zip(ls, count)]
                )
            )
            n_shells_per_atom.append(np.sum(count))

        self.l_per_basis_func = np.concatenate(basis_func_to_l_per_atom_type)
        self.m_per_basis_func = np.concatenate(basis_func_to_m_per_atom_type)
        assert len(self.l_per_basis_func) == len(self.m_per_basis_func) == self.n_basis

        self.n_shells_per_atom = np.array(n_shells_per_atom)

        self.shell_beginning_mask = np.equal(self.l_per_basis_func, -self.m_per_basis_func)
        self.basis_func_to_shell = np.cumsum(self.shell_beginning_mask) - 1
        self.shell_to_first_basis_func = self.shell_beginning_mask.nonzero()[0]
        self.l_per_shell = self.l_per_basis_func[self.shell_to_first_basis_func]
        self.m_per_shell = self.m_per_basis_func[self.shell_to_first_basis_func]

        self.check_consistency()

    @classmethod
    def get_irreps_and_integrals(
        cls, one_atom_mols: list[pyscf.gto.M]
    ) -> tuple[dict[str, list], list[Irreps], list[np.ndarray]]:
        """Get the irreps and integrals for a list of molecules with one atom each.

        Args:
            one_atom_mols: List of molecules with one atom each.

        Returns:
            List of irreps for each atom.
            List of integrals for each atom.
            Dict of basis sets for each atom.
        """
        irreps_per_atom = []
        basis_dict = {}
        for mol in one_atom_mols:
            basis_dict[mol.atom_symbol(0)] = mol._basis[mol.atom_symbol(0)]
            # read ls (rotation orders) from the mol object
            ls = [mol.bas_angular(i) for i in range(mol.nbas) for _ in range(mol.bas_nctr(i))]
            assert np.all(
                ls[:-1] <= ls[1:]
            ), f"orbitals are expected to be sorted by rotation order l, but got order {ls}"

            # group ls, count multiplicities
            ls, multiplicities = np.unique(ls, return_counts=True)

            # create irreps, using 'y' parity (1 for even l, -1 for odd l)
            irreps_per_atom.append(
                Irreps("+".join([f"{mult}x{l}y" for mult, l in zip(multiplicities, ls)]))
            )

        basis_integrals = [
            get_normalization_vector(one_atom_mol, clip_minimum=1.6e-15)
            for one_atom_mol in one_atom_mols
        ]

        return basis_dict, irreps_per_atom, basis_integrals

    @classmethod
    def from_nwchem(
        cls, nwchem_str: str, atomic_numbers: np.ndarray | list[int] = None
    ) -> "BasisInfo":
        """Construct a BasisInfo object from a string in nwchem format.

        Args:
            nwchem_str: String containing the basis in nwchem format.
            atomic_numbers: List of atomic numbers of the atoms in the basis. If None, uses all atomic
                numbers present in the NWChem string.
        """
        # create a list of the basis sets for each atom, in the internal pyscf format
        one_atom_mols = []

        loaded_atomic_numbers = []
        # iterate over all (or the requested if specified) elements and try to find them in the nwchem string
        for atomic_number in (
            np.arange(len(pyscf.lib.param.ELEMENTS) - 1)
            if atomic_numbers is None
            else atomic_numbers
        ):
            symbol = pyscf.lib.param.ELEMENTS[atomic_number]
            try:
                basis_for_element = pyscf.gto.parse(nwchem_str, symb=symbol, optimize=True)
                loaded_atomic_numbers.append(atomic_number)
                one_atom_mols.append(
                    pyscf.gto.M(
                        atom=f"{symbol} 0 0 0",
                        spin=None,
                        basis={symbol: basis_for_element},
                    )
                )
            except pyscf.gto.BasisNotFoundError:
                # if an element is not found, continue
                continue

        assert len(one_atom_mols) == len(loaded_atomic_numbers)
        assert (
            len(loaded_atomic_numbers) > 0
        ), f"no atomic numbers successfully loaded from nwchem string {nwchem_str}."
        if atomic_numbers is not None:
            missing_atomic_numbers = sorted(list(set(atomic_numbers) - set(loaded_atomic_numbers)))
            missing_atomic_symbols = [
                pyscf.lib.param.ELEMENTS[atomic_number] for atomic_number in missing_atomic_numbers
            ]
            assert len(loaded_atomic_numbers) == len(atomic_numbers), (
                f"atomic numbers successfully loaded from nwchem string {loaded_atomic_numbers} do not match requested "
                f"atomic numbers {atomic_numbers}, as they are missing {missing_atomic_numbers}, "
                f"i.e. elements {missing_atomic_symbols}"
            )
        loaded_atomic_numbers = np.array(loaded_atomic_numbers, dtype=np.uint8)

        basis_dict, irreps_per_atom, basis_integrals = cls.get_irreps_and_integrals(one_atom_mols)
        return cls(
            atomic_numbers=loaded_atomic_numbers,
            basis_dict=basis_dict,
            irreps_per_atom=irreps_per_atom,
            integrals=basis_integrals,
        )

    @classmethod
    def from_atomic_numbers_with_even_tempered_basis(
        cls,
        atomic_numbers: list[int] | np.ndarray,
        basis: str = "6-31G(2df,p)",
        beta: float = 2.5,
    ) -> "BasisInfo":
        """Construct a BasisInfo object from a list of atomic numbers, using an even-tempered
        basis. The list or array of atomic numbers should have unique values!

        Args:
            atomic_numbers: List of atomic numbers of the atoms in the basis.
            basis: Basis set to use before converting to even-tempered basis. This matters and should be the same as was
                used during Kohn-Sham calculations. Defaults to "6-31G(2df,p)".
            beta: Exponent factor :math:`\\beta` of the even-tempered basis set. Defaults to 2.5.

        Returns:
            BasisInfo object.
        """
        assert len(atomic_numbers) == len(
            set(atomic_numbers)
        ), "atomic_numbers values must be unique."
        atomic_numbers = np.array(atomic_numbers, dtype=np.uint8)
        one_atom_mols = []
        for i in atomic_numbers:
            # The basis matters so need to be careful which one to use
            one_atom_mol_tmp = pyscf.gto.M(atom=f"{i} 0 0 0", spin=None, basis=basis)
            one_atom_mol = build_mol_with_even_tempered_basis(
                one_atom_mol_tmp, beta=beta, spin=None
            )
            one_atom_mols.append(one_atom_mol)
        basis_dict, irreps_per_atom, basis_integrals = cls.get_irreps_and_integrals(one_atom_mols)
        return cls(
            atomic_numbers=atomic_numbers,
            basis_dict=basis_dict,
            irreps_per_atom=irreps_per_atom,
            integrals=basis_integrals,
        )

    @classmethod
    def from_dataset_info_yaml(
        cls, path_to_data_info: str, atomic_numbers: np.ndarray
    ) -> "BasisInfo":
        """Construct a BasisInfo object from a yaml file corresponding to a dataset. Can create the
        basis info object corresponding to an even- tempered or any other basis set.

        Args:
            path_to_data_info: The path to create the hydra config for this dataset.
            atomic_numbers: List of atomic numbers of the atoms in the basis.

        Returns:
            BasisInfo object.
        """
        if not os.path.isfile(path_to_data_info):
            logger.warning(
                "The dataset info file was not found. Assuming that an even-tempered basis set with beta=2.5 was used."
            )
            return BasisInfo.from_atomic_numbers_with_even_tempered_basis(atomic_numbers)
        cfg = OmegaConf.load(path_to_data_info)
        kohn_sham_datagen_info = cfg.get("kohn_sham")
        kohn_sham_basis = kohn_sham_datagen_info["basis"]

        of_basis_set_name = cfg.get("of_basis_set")
        assert len(atomic_numbers) == len(
            set(atomic_numbers)
        ), "atomic_numbers values must be unique."
        atomic_numbers = np.array(atomic_numbers, dtype=np.uint8)
        one_atom_mols = []
        for i in atomic_numbers:
            # The basis matters so need to be careful which one to use
            one_atom_mol_tmp = pyscf.gto.M(atom=f"{i} 0 0 0", spin=None, basis=kohn_sham_basis)
            one_atom_mol = construct_aux_mol(one_atom_mol_tmp, of_basis_set_name, spin=None)
            one_atom_mols.append(one_atom_mol)
        basis_dict, irreps_per_atom, basis_integrals = cls.get_irreps_and_integrals(one_atom_mols)
        return cls(
            atomic_numbers=atomic_numbers,
            basis_dict=basis_dict,
            irreps_per_atom=irreps_per_atom,
            integrals=basis_integrals,
        )

    @classmethod
    def from_mol(cls, mol: pyscf.gto.Mole):
        """Construct a BasisInfo object from a molecule (not suited for ML functionals).

        Uses the basis of the molecule as is. Only the basis functions of the elements actually
        present in the molecule will be included in the BasisInfo.

        Args:
            mol: Molecule object.

        Returns:
            BasisInfo object.
        """
        atomic_numbers = np.array(mol.atom_charges(), dtype=np.uint8)
        atomic_numbers = np.unique(atomic_numbers)

        one_atom_mols = []
        for i in atomic_numbers:
            # The basis matters so need to be careful which one to use
            one_atom_mol = pyscf.gto.M(atom=f"{i} 0 0 0", spin=None, basis=mol.basis)
            one_atom_mols.append(one_atom_mol)
        basis_dict, irreps_per_atom, basis_integrals = cls.get_irreps_and_integrals(one_atom_mols)
        return cls(
            atomic_numbers=atomic_numbers,
            basis_dict=basis_dict,
            irreps_per_atom=irreps_per_atom,
            integrals=basis_integrals,
        )

    def check_consistency(self) -> None:
        """Check whether the BasisInfo object is consistent."""

        assert len(self.atomic_numbers) == len(self.irreps_per_atom) == len(self.integrals), (
            f"atomic_numbers, irreps_per_atom and integrals must have the same length, "
            f"but got {len(self.atomic_numbers)}, {len(self.irreps_per_atom)} and {len(self.integrals)} respectively."
        )

        for irreps, integrals in zip(self.irreps_per_atom, self.integrals):
            assert irreps.dim == len(integrals), (
                f"irreps and integrals must have the same dimensionality, "
                f"but got {irreps.dim} and {len(integrals)} respectively."
            )

    @property
    def n_basis(self) -> int:
        """Total number of basis functions."""
        return np.sum(self.basis_dim_per_atom)

    @property
    def n_shells(self) -> int:
        """Total number of shells."""
        return np.sum(self.n_shells_per_atom)

    @property
    def n_types(self) -> int:
        """Number of atom types."""
        return len(self.basis_dim_per_atom)

    def split_field_by_atom_type(
        self, array_or_tensor: np.ndarray | torch.Tensor, axis: int = 0
    ) -> tuple[np.ndarray | torch.Tensor, ...]:
        """Split the given array or tensor into parts corresponding to different atom types.

        Args:
            array_or_tensor: Array or tensor to split.
            axis: Axis along which to split.

        Returns:
            tuple of arrays or tensors, one for each atom.
        """

        assert (
            array_or_tensor.shape[axis] == self.n_basis
        ), f"array_or_tensor.shape[axis] must be {self.n_basis=}, but is {array_or_tensor.shape[axis]}"
        if isinstance(array_or_tensor, torch.Tensor):
            return torch.split(array_or_tensor, list(self.basis_dim_per_atom), dim=axis)
        if isinstance(array_or_tensor, np.ndarray):
            return tuple(
                np.split(array_or_tensor, np.cumsum(self.basis_dim_per_atom)[:-1], axis=axis)
            )
        raise ValueError(
            f"array_or_tensor must be of type np.ndarray or torch.Tensor, but is {type(array_or_tensor)}"
        )

    def __eq__(self, other: object) -> bool:
        """Check whether two BasisInfo objects are equal."""
        if not isinstance(other, BasisInfo):
            return False
        if (
            not np.allclose(
                self.atomic_numbers,
                other.atomic_numbers,
            )
            or not np.allclose(
                self.atomic_number_to_atom_index,
                other.atomic_number_to_atom_index,
            )
            or not np.allclose(
                self.atomic_number_to_atom_index,
                other.atomic_number_to_atom_index,
            )
            or not self.basis_dict == other.basis_dict
            or not np.allclose(
                self.basis_dim_per_atom,
                other.basis_dim_per_atom,
            )
            or not np.allclose(
                self.l_per_basis_func,
                other.l_per_basis_func,
            )
            or not np.allclose(
                self.m_per_basis_func,
                other.m_per_basis_func,
            )
            or not self.n_basis == other.n_basis
            or not self.n_types == other.n_types
        ):
            return False

        for i, integrals in enumerate(self.integrals):
            if not np.allclose(integrals, other.integrals[i]):
                return False
        for i, irrep in enumerate(self.irreps_per_atom):
            if not irrep == other.irreps_per_atom[i]:
                return False
        for i, basis_func_indx in enumerate(self.atom_ind_to_basis_function_ind):
            if not np.allclose(
                basis_func_indx,
                other.atom_ind_to_basis_function_ind[i],
            ):
                return False

        return True

    def is_subset(self, other: "BasisInfo") -> bool:
        """Check whether the BasisInfo object is a subset of another BasisInfo object.

        Args:
            other: BasisInfo object to check against.

        Returns:
            True if the BasisInfo object is a subset of the other BasisInfo
            object.
        """
        if (
            not np.all(np.in1d(self.atomic_numbers, other.atomic_numbers))
            or not self.n_basis <= other.n_basis
            or not self.n_types <= other.n_types
        ):
            return False

        for atom_type, basis in self.basis_dict.items():
            if basis != other.basis_dict[atom_type]:
                return False

        return True
