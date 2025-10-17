"""Definition of the :class:`OFData` class, whose instances are the inputs to the model."""

from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr
from loguru import logger
from pyscf import gto
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_add_pool

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ofdft.basis_integrals import get_normalization_vector
from mldft.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class StrEnumwithCheck(StrEnum):
    """Enum with a check if a value is in the enum."""

    @classmethod
    def check_key(cls, value):
        """Check if a value is in the enum.

        If not this will throw a value error.
        """
        cls(value)


class Representation(StrEnumwithCheck):
    """Enum for the representations of the fields in the OFData object.

    *  NONE: Quantities that are no geometric objects and are thus not transformed
    *  SCALAR: Scalar quantities (e.g. energies). Stay constant under basis transformations.
    *  VECTOR: Vector quantities that transform with the basis transformation matrix.
    *  DUAL_VECTOR: Vectors that are applied to vectors using the scalar product. Gradients are dual vectors, but we
        differentiate between gradients and dual vectors because the `ProjectGradients` is only applied to gradients.
    *  GRADIENT: Dual vectors that are gradients and as such can be projected to the electron preserving manifold.
    *  ENDOMORPHISM: Endomorphisms that are applied to vectors.
    *  BILINEAR_FORM: Bilinear forms that are applied to vectors.
    *  AO: Atomic orbitals.

    For exact transformation formulas see :func:`~mldft.ml.data.components.basis_transforms.transform_tensor`.
    """

    NONE = "none"
    SCALAR = "scalar"  #
    VECTOR = "vector"
    # Gradients are dual vectors and transform the same. We differentiate because the `ProjectGradients` is only applied
    # to gradients and not dual vectors (i.e. the dual_basis_integrals)
    GRADIENT = "gradient"
    DUAL_VECTOR = "dual_vector"
    ENDOMORPHISM = "endomorphism"
    BILINEAR_FORM = "bilinear_form"
    AO = "ao"


REPRESENTATION_DICT = dict(
    coeffs=Representation.VECTOR,
    dual_basis_integrals=Representation.DUAL_VECTOR,
    grad_kin=Representation.GRADIENT,
    grad_hartree=Representation.GRADIENT,
    grad_xc=Representation.GRADIENT,
    grad_ext=Representation.GRADIENT,
    grad_ext_mod=Representation.GRADIENT,
    grad_kinapbe=Representation.GRADIENT,
    grad_kin_minus_apbe=Representation.GRADIENT,
    grad_kin_plus_xc=Representation.GRADIENT,
    grad_tot=Representation.GRADIENT,
)


class OFData(Data):
    """Pytorch geometric data object for OF-DFT data. By pytorch-geometric magic (dynamic
    inheritance), the properties and methods of :class:`OFData` are also available for
    :class:`torch_geometric.data.Batch` objects constructed out of them.
    Attributes:
        pos: Atomic positions. Shape (n_atom, 3).
        atomic_numbers: Atomic numbers. Shape (n_atom,).
        atom_ind: Indices of the atoms in the basis. Shape (n_atom,).
        n_atom: Number of atoms in the molecule.
        n_basis: Number of basis functions.
        n_electron: Number of electrons in the molecule.
        n_basis_per_atom: Number of basis functions per atom. Shape (n_atom,). Can be used with :func:`torch.split`.
        atom_ptr: Pointer tensor that can be used to split basis functions into atoms using :func:`np.split`.
            See `OFData.split_field_by_atom`. Shape (n_atom,).
        basis_function_ind: Array holding the data fields basis function indices. Can be used to
            map atomic number specific quantities to the data fields. Shape (n_basis,).
        coeff_ind_to_node_ind: Array mapping coefficient indices to node (=atom) indices (we use 'node' instead of
            'atom' to avoid confusion with :attr:`atom_ind`, which indexes different atomic numbers). Shape (n_basis,).
        dual_basis_integrals: Dual basis integrals, required to compute the projected gradient. Shape (n_basis,).
        coeffs: Density coefficients in the linear OF Ansatz. Shape (n_basis,).
    .. note::
        Optional Attributes: (might not be present or None)
            * ``irreps_per_atom``: Irreps of the basis functions per atom. Shape (n_atom,).
            * ``ground_state_coeffs``: Ground-state density coefficients in the linear OF Ansatz. Shape (n_basis,).
            * ``gradient_label``: Gradient of the (by default) kinetic energy w.r.t. the coefficients. Shape (n_basis,).
            * ``energy_label``: Total energy of the molecule.
            * ``has_energy_label``: Whether the energy label is present. (False for data from the initialisation.)
            * ``mol_id``: ID of the molecule.
            * ``scf_iteration``: Index of the SCF iteration.
            * ``atom_coo_indices``: Indices of the basis functions in the molecule. Shape (2, n_basis).
              Only present if :py:func:`~mldft.ml.data.components.convert_transforms.add_atom_coo_indices` was applied.
    """

    @classmethod
    def construct_new(
        cls,
        basis_info: BasisInfo,
        pos: np.ndarray,
        atomic_numbers: np.ndarray,
        coeffs: np.ndarray,
        ground_state_coeffs: np.ndarray = None,
        gradient_label: np.ndarray = None,
        energy_label: list[float] | np.ndarray = None,
        has_energy_label: bool = None,
        dual_basis_integrals: np.ndarray | str = None,
        add_irreps: bool = False,
        mol_id: str = None,
        scf_iteration: int = None,
        additional_representations: dict[str, Representation] = None,
        **kwargs,
    ) -> "OFData":
        """Construct a new OFData object from the given data, inferring additional fields using the
        given BasisInfo. We do not override the __init__ because that messes with pytorch geometric
        magic (somewhere in the loader).

        Args:
            basis_info: Basis information for the data.
            pos: Atomic positions. Shape (n_atom, 3).
            atomic_numbers: Atomic numbers. Shape (n_atom,).
            coeffs: Density coefficients in the linear OF Ansatz. Shape (n_basis,).
            ground_state_coeffs: Ground-state density coefficients in the linear OF Ansatz. Shape (n_basis,). Optional.
            gradient_label: Gradient of the (by default) kinetic energy w.r.t. the coefficients. Shape (n_basis,). Optional.
            energy_label: Total energy of the molecule. Optional.
            has_energy_label: Whether the energy label is present. (False for data from the initialisation.) Optional.
            dual_basis_integrals: Dual basis integrals, required to compute the projected gradient. If set to `infer_from_basis`, it will be computed from the basis and be in the untransformed basis.
            add_irreps: Whether to add the irreps of the basis functions per atom to the sample.
            mol_id: ID of the molecule. Optional.
            scf_iteration: Index of the SCF iteration. Optional.
            additional_representations: Dictionary specifying the transformation order of the data fields.
            kwargs: Additional keyword arguments to be passed to the init of :class:`Data` from pytorch_geometric.
        """

        # verify that the shapes are correct
        assert isinstance(
            basis_info, BasisInfo
        ), f"basis_info must be of type BasisInfo, but is {type(basis_info)}"
        n_basis = coeffs.shape[0]
        n_atom = atomic_numbers.shape[0]
        assert pos.shape == (
            atomic_numbers.shape[0],
            3,
        ), f"pos must have shape (n_atom, 3), but has shape {pos.shape}"
        assert atomic_numbers.shape == (
            n_atom,
        ), f"atomic_numbers must have shape (n_atom,), but has shape {atomic_numbers.shape}"
        assert coeffs.shape == (
            n_basis,
        ), f"coeffs must have shape (n_basis,), but has shape {coeffs.shape}"

        if ground_state_coeffs is not None:
            assert ground_state_coeffs.shape == (n_basis,)
        if gradient_label is not None:
            assert gradient_label.shape == (n_basis,)
        if energy_label is not None:
            energy_label = np.asarray(energy_label)
            assert energy_label.shape == (1,)
        if has_energy_label is not None:
            has_energy_label = np.asarray(has_energy_label)
            assert has_energy_label.shape == (1,)

        # add additional fields, inferred using the basis info

        # this is a list that can be indexed with the atomic number to get the index of that atomic number in the
        # basis.
        # e.g. if oxygen would be the third-lightest element described in the basis,
        # then atomic_number_to_atom_index[8] == 2.
        # atomic numbers not present in the basis are assigned the index -1.
        atomic_number_to_atom_index = basis_info.atomic_number_to_atom_index

        # we cannot call it "atom_index" because attributes having "index" as a substring
        # are treated specially by pytorch geometric
        atom_ind = atomic_number_to_atom_index[atomic_numbers]
        assert np.all(atom_ind >= 0), (
            f"Some atoms have no basis functions assigned to them: Atomic numbers "
            f"{np.unique(atomic_numbers[atom_ind == -1])}."
        )

        # after casting to a tuple, this can be used with torch.split,
        # e.g. torch.split(data.coeffs, tuple(data.n_basis_per_atom))
        n_basis_per_atom = basis_info.basis_dim_per_atom[atom_ind]

        # could remove atom_ptr completely in favour of result.n_basis_per_atom,
        # also alleviating the need for a custom __inc__ as implemented in OFData
        atom_ptr = np.cumsum(n_basis_per_atom)
        basis_function_ind = np.concatenate(
            [basis_info.atom_ind_to_basis_function_ind[i] for i in atom_ind]
        )

        coeff_ind_to_node_ind = np.repeat(np.arange(len(n_basis_per_atom)), n_basis_per_atom)
        if isinstance(dual_basis_integrals, str) and dual_basis_integrals == "infer_from_basis":
            dual_basis_integrals = np.concatenate(basis_info.integrals[atom_ind])
        assert (
            dual_basis_integrals is not None
        ), "dual_basis_integrals must be provided. Are you using very old data?"

        if add_irreps:
            irreps_per_atom = basis_info.irreps_per_atom[atom_ind]
        else:
            irreps_per_atom = None

        representations = {
            "pos": Representation.NONE,
            "atomic_numbers": Representation.NONE,
            "coeffs": Representation.VECTOR,
            "ground_state_coeffs": Representation.VECTOR,
            "gradient_label": Representation.GRADIENT,
            "energy_label": Representation.SCALAR,
            "has_energy_label": Representation.NONE,
            "atom_ind": Representation.NONE,
            "n_basis_per_atom": Representation.NONE,
            "atom_ptr": Representation.NONE,
            "basis_function_ind": Representation.NONE,
            "coeff_ind_to_node_ind": Representation.NONE,
            "dual_basis_integrals": Representation.DUAL_VECTOR,
            "mol_id": Representation.NONE,
            "scf_iteration": Representation.NONE,
            "irreps_per_atom": Representation.NONE,
            "representations": Representation.NONE,
        }
        if additional_representations is not None:
            representations.update(additional_representations)

        result = cls(
            pos=pos,
            atomic_numbers=atomic_numbers,
            coeffs=coeffs,
            ground_state_coeffs=ground_state_coeffs,
            gradient_label=gradient_label,
            energy_label=energy_label,
            has_energy_label=has_energy_label,
            atom_ind=atom_ind,
            n_basis_per_atom=n_basis_per_atom,
            atom_ptr=atom_ptr,
            basis_function_ind=basis_function_ind,
            coeff_ind_to_node_ind=coeff_ind_to_node_ind,
            dual_basis_integrals=dual_basis_integrals,
            mol_id=mol_id,
            scf_iteration=scf_iteration,
            irreps_per_atom=irreps_per_atom,
            representations=representations,
            **kwargs,
        )
        return result

    """Pytorch geometric data object for OF-DFT data. By pytorch-geometric magic (dynamic
    inheritance), the properties and methods of :class:`OFData` are also available for
    :class:`torch_geometric.data.Batch` objects constructed out of them.

    Attributes:
        pos: Atomic positions. Shape (n_atom, 3).
        atomic_numbers: Atomic numbers. Shape (n_atom,).
        atom_ind: Indices of the atoms in the basis. Shape (n_atom,).
        n_atom: Number of atoms in the molecule.
        n_basis: Number of basis functions.
        n_electron: Number of electrons in the molecule.
        n_basis_per_atom: Number of basis functions per atom. Shape (n_atom,). Can be used with :func:`torch.split`.
        atom_ptr: Pointer tensor that can be used to split basis functions into atoms using :func:`np.split`.
            See `OFData.split_field_by_atom`. Shape (n_atom,).
        basis_function_ind: Array holding the data fields basis function indices. Can be used to
            map atomic number specific quantities to the data fields. Shape (n_basis,).
        coeff_ind_to_node_ind: Array mapping coefficient indices to node (=atom) indices (we use 'node' instead of
            'atom' to avoid confusion with :attr:`atom_ind`, which indexes different atomic numbers). Shape (n_basis,).
        dual_basis_integrals: Dual basis integrals, required to compute the projected gradient when using non-orthogonal
            basis_transformations. Shape (n_basis,).
        coeffs: Density coefficients in the linear OF Ansatz. Shape (n_basis,).

    .. note::
        Optional Attributes: (might not be present or None)
            * ``irreps_per_atom``: Irreps of the basis functions per atom. Shape (n_atom,).
            * ``ground_state_coeffs``: Ground-state density coefficients in the linear OF Ansatz. Shape (n_basis,).
            * ``gradient_label``: Gradient of the (by default) kinetic energy w.r.t. the coefficients. Shape (n_basis,).
            * ``energy_label``: Total energy of the molecule.
            * ``has_energy_label``: Whether the energy label is present. (False for data from the initialisation.)
            * ``mol_id``: ID of the molecule.
            * ``scf_iteration``: Index of the SCF iteration.
            * ``atom_coo_indices``: Indices of the basis functions in the molecule. Shape (2, n_basis).
              Only present if :py:func:`~mldft.ml.data.components.convert_transforms.add_atom_coo_indices` was applied.
    """

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        scf_iteration: int,
        basis_info: "BasisInfo",
        energy_key: str = "e_kin",
        gradient_key: str = "grad_kin",
        add_irreps: bool = False,
        additional_keys_at_scf_iteration: dict[str, Representation] = None,
        additional_keys_at_ground_state: dict[str, Representation] = None,
        additional_keys_per_geometry: dict[str, Representation] = None,
    ) -> "OFData":
        """Load a sample from the DFT data set.

        Args:
            path: Path to the .zarr file.
            scf_iteration: Index of the SCF iteration to load.
            basis_info: :class:`~mldft.ml.data.components.basis_info.BasisInfo` object containing information about the
                OF basis.
            energy_key: Key of the energy to load.
            gradient_key: Key of the gradient to load.
            add_irreps: Whether to add the irreps of the basis functions per atom to the sample.
            additional_keys_at_scf_iteration: List of additional keys to load from the zarr group.
                The arrays corresponding to these keys will be indexed at the current SCF iteration. Optional.
            additional_keys_at_ground_state: List of additional keys to load from the zarr group.
                The arrays corresponding to these keys will be indexed at the final SCF iteration,
                i.e. the ground state. Optional.
            additional_keys_per_geometry: List of additional keys to load from the zarr group.
                The arrays corresponding to these keys will not be indexed. Optional.

        Returns:
            sample: pytorch geometric data object. It has the following fields:
                - pos: Atomic positions. Shape (n_atom, 3).
                - atomic_numbers: Atomic numbers. Shape (n_atom,).
                - atom_ind: Indices of the atoms in the basis. Shape (n_atom,).
                - n_basis_per_atom: Number of basis functions per atom. Shape (n_atom,).
                  Can be used with :func:`torch.split`.
                - atom_ptr: Indices of the atoms that belong to the molecule. Shape (n_atom,).
                  Can be used with :func:`np.split`.
                - coeffs: Density coefficients in the linear OF Ansatz. Shape (n_basis,).
                - ground_state_coeffs: Ground-state density coefficients in the linear OF Ansatz. Shape (n_basis,).
                - gradient_label: Gradient of the (by default) kinetic energy w.r.t. the coefficients. Shape (n_basis,).
                - energy_label: Total energy of the molecule.
                - has_energy_label: Whether the energy/gradient label is meaningful. (set to 0 for data from the initialisation.)
        """
        root = zarr.open(path, mode="r")
        geometry = root["geometry"]
        of_labels = root["of_labels"]

        if scf_iteration < 0:
            num_scf_iterations = of_labels["n_scf_steps"][()]
            scf_iteration = int(num_scf_iterations + scf_iteration)

        if "dual_basis_integrals" in of_labels["spatial"]:
            dual_basis_integrals = np.asarray(
                of_labels["spatial"]["dual_basis_integrals"][scf_iteration]
            )
        else:
            dual_basis_integrals = None

        # Check for additional keys to load
        additional_keys = {}
        additional_key_representations = {}

        if additional_keys_at_scf_iteration is not None:
            for key, representation in additional_keys_at_scf_iteration.items():
                Representation.check_key(representation)
                additional_keys[key] = root[key][scf_iteration]
                additional_key_representations[key] = representation

        if additional_keys_at_ground_state is not None:
            for key, representation in additional_keys_at_ground_state.items():
                Representation.check_key(representation)
                # add prefix "ground_state_" after last slash in key name
                split = key.rsplit("/", 1)
                try:
                    additional_keys[split[0] + "/ground_state_" + split[1]] = root[key][-1]
                    additional_key_representations[
                        split[0] + "/ground_state_" + split[1]
                    ] = representation
                except KeyError:
                    pass

        if additional_keys_per_geometry is not None:
            for key, representation in additional_keys_per_geometry.items():
                Representation.check_key(representation)
                additional_keys[key] = root[key][()]
                additional_key_representations[key] = representation

        if "has_energy_label" not in root["ks_labels"]["energies"]:
            has_energy_label = [True]
        else:
            has_energy_label = [root["ks_labels"]["energies"]["has_energy_label"][scf_iteration]]

        result = cls.construct_new(
            basis_info=basis_info,
            add_irreps=add_irreps,
            mol_id=geometry["mol_id"][()],
            scf_iteration=scf_iteration,
            pos=geometry["atom_pos"][:],
            atomic_numbers=geometry["atomic_numbers"][:],
            coeffs=of_labels["spatial"]["coeffs"][scf_iteration],
            ground_state_coeffs=of_labels["spatial"]["coeffs"][-1],
            gradient_label=of_labels["spatial"][gradient_key][scf_iteration],
            energy_label=[of_labels["energies"][energy_key][scf_iteration]],
            has_energy_label=has_energy_label,
            dual_basis_integrals=dual_basis_integrals,
            additional_representations=additional_key_representations,
            **additional_keys,
        )

        return result

    @classmethod
    def from_file_with_all_gradients(
        cls,
        path: str | Path,
        scf_iteration: int,
        basis_info: "BasisInfo",
        energy_key: str = "e_kin",
        gradient_key: str = "grad_kin",
        add_irreps: bool = True,
    ) -> "OFData":
        """Load a sample containing all the keys in the zarr file."""
        root = zarr.open(path, mode="r")
        geometry = root["geometry"]
        of_labels = root["of_labels"]
        of_spatial = of_labels["spatial"]

        spatial_keys = list(of_spatial.keys())
        missing_keys = set(spatial_keys).difference(REPRESENTATION_DICT.keys())
        assert not missing_keys, (
            f"Keys {list(missing_keys)} missing from REPRESENTATION_DICT. "
            f"Please add them, with their corresponding representations."
        )
        additional_keys = spatial_keys.copy()
        # The coeffs are already set in the construct_new, so we need to exclude them
        additional_keys.remove("coeffs")
        additional_arrays = {f"{key}": of_spatial[key][scf_iteration] for key in additional_keys}
        dual_basis_integrals = additional_arrays.pop("dual_basis_integrals")
        additional_representations = {
            key: REPRESENTATION_DICT[key] for key in additional_arrays.keys()
        }

        if "has_energy_label" not in root["ks_labels"]["energies"]:
            has_energy_label = [True]
            # could issue a warning but spam
        else:
            has_energy_label = [root["ks_labels"]["energies"]["has_energy_label"][scf_iteration]]
        result = cls.construct_new(
            basis_info=basis_info,
            add_irreps=add_irreps,
            mol_id=geometry["mol_id"][()],
            scf_iteration=scf_iteration,
            pos=geometry["atom_pos"][:],
            atomic_numbers=geometry["atomic_numbers"][:],
            coeffs=of_labels["spatial"]["coeffs"][scf_iteration],
            ground_state_coeffs=of_labels["spatial"]["coeffs"][-1],
            gradient_label=of_labels["spatial"][gradient_key][scf_iteration],
            energy_label=[of_labels["energies"][energy_key][scf_iteration]],
            has_energy_label=has_energy_label,
            dual_basis_integrals=dual_basis_integrals,
            additional_representations=additional_representations,
            **additional_arrays,
        )
        return result

    @classmethod
    def minimal_sample_from_mol(
        cls,
        mol: gto.Mole,
        basis_info: BasisInfo | None = None,
        add_transformation_matrix: bool = False,
    ) -> "OFData":
        """Construct an OFData sample from a molecule.

        Used for running (classical) OFDFT calculation on a molecule from scratch.
        Assumes the basis of the molecule is the one used for the OFDFT calculation and the basis
        functions present in the molecule are sufficient information (as is the case for classical
        ofdft but unlikely in case of ML functionals).

        Args:
            mol: The molecule.
            basis_info: Basis information for the data. If None, a minimal basis info object is
                created.
            add_transformation_matrix: Whether to add identity matrices for the transformation
                and inverse transformation matrices. Default is False.

        Returns:
            OFData: The OFData sample.
        """
        if basis_info is None:
            basis_info = BasisInfo.from_mol(mol)
            logger.warning(
                "Beware of use with ML functionals, the basis info object is minimal and likely "
                "not compatible with ML functionals!"
            )

        sample = cls.construct_new(
            basis_info=basis_info,
            pos=mol.atom_coords(),
            atomic_numbers=mol.atom_charges(),
            coeffs=np.zeros(mol.nao),
            add_irreps=basis_info is not None,
            dual_basis_integrals=get_normalization_vector(mol),
        )

        if add_transformation_matrix:
            transform = np.eye(sample.n_basis, dtype=np.float64)
            inv_transform = np.eye(sample.n_basis, dtype=np.float64)
            sample.add_item("transformation_matrix", transform, Representation.VECTOR)
            sample.add_item("inv_transformation_matrix", inv_transform, Representation.DUAL_VECTOR)

        return sample

    def add_item(self, key: str, value: Any, representation: Representation | str):
        """Add an item to the data object. The transformation order specifies how the item should
        be transformed.

        Args:
            key: Key of the item to add.
            value: Value of the item to add.
            representation: Transformation order of the item.
        """
        if key.startswith("_"):
            log.warning(
                f"Adding item {key} to OFData with leading underscore. This will not be accessible using dot notation"
                f"or .keys()."
            )
        Representation.check_key(representation)
        self.representations[key] = representation
        setattr(self, key, value)

    def delete_item(self, key: str) -> None:
        """Remove an item from the data object.

        Args:
            key: Key of the item to remove.
        """
        delattr(self, key)
        self.representations.pop(key)

    def split_field_by_atom(
        self, array_or_tensor: np.ndarray | torch.Tensor, axis: int = 0
    ) -> tuple[np.ndarray | torch.Tensor, ...]:
        """Split the given array or tensor into the individual atoms. The array or tensor is not
        required to be part of the OFData object, and if it is, the corresponding attribute will
        not be overridden.

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
            return torch.split(
                array_or_tensor, self.n_basis_per_atom.cpu().int().tolist(), dim=axis
            )
        if isinstance(array_or_tensor, np.ndarray):
            return tuple(np.split(array_or_tensor, self.atom_ptr[:-1], axis=axis))
        raise ValueError(
            f"array_or_tensor must be of type np.ndarray or torch.Tensor, but is {type(array_or_tensor)}"
        )

    @property
    def n_basis(self) -> int:
        # Number of basis functions of the molecule(s)
        return self.coeffs.shape[0]

    @property
    def n_atom(self) -> int:
        # Number of atoms in the molecule(s)
        return self.pos.shape[0]

    @property
    def n_electron(self) -> int:
        # Number of electrons in the molecule(s)
        return self.atomic_numbers.sum().item()

    def get_n_atom_per_molecule(self):
        """Get the number of atoms per molecule.

        Only available for Batch objects.
        """
        assert isinstance(
            self, Batch
        ), "get_n_atoms_per_molecule is only available for Batch objects"
        return global_add_pool(torch.ones_like(self.atomic_numbers), self.batch)

    def get_n_basis_per_molecule(self):
        """Get the number of basis functions per molecule.

        Only available for Batch objects.
        """
        assert isinstance(
            self, Batch
        ), "get_n_basis_per_molecule is only available for Batch objects"
        return global_add_pool(self.n_basis_per_atom, self.batch)

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        """Increment the attribute `key` by `value` when creating a batch. We change the default
        behavior for two keys, "atom_ptr" and "atom_coo_indices", both of which consist of indices
        of different basis functions. Hence, they must be incremented by the number of basis
        functions to correctly reference basis functions in the batch. Furthermore, we need to
        increment "coeff_ind_to_node_ind" by the number of atoms in the batch, because its values
        are node (=atom) indices.

        See the docstring of :class:`torch_geometric.data.Data` for general information on :meth:`__inc__`.
        """
        if key in ("atom_ptr", "atom_coo_indices"):
            return self.n_basis
        elif key == "coeff_ind_to_node_ind":
            # this is a mapping from coefficient indices to node (=atom) indices
            # we need to increment it by the number of atoms in the batch
            return self.n_atom
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        """Concatenate the attribute `key` along dimension `value` when creating a batch.

        We change the default behavior for the key "atom_coo_indices", of shape (2, n_basis), which
        should be concatenated along the first dimension.
        """
        if key == "atom_coo_indices":
            return 1
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)

    def __setattr__(self, key: str, value: Any):
        """Overwrite the default ``__setattr__`` to check if fields are added without their
        transformation order being set."""
        propobj = getattr(self.__class__, key, None)
        if propobj is not None and getattr(propobj, "fset", None) is not None:
            propobj.fset(self, value)
        else:
            # if the attribute already exists, we do not need to warn
            if not hasattr(self, key):
                is_batch_ptr = isinstance(self, Batch) and (
                    key.endswith("ptr") or key.endswith("batch")
                )
                key_to_check = key if not is_batch_ptr else "_".join(key.split("_")[:-1])
                # Some fields (like pos) are added by pytorch_geometric before the representations are set.
                if (
                    hasattr(self, "representations")
                    and key_to_check not in self.representations.keys()
                ):
                    # we ignore private attributes since they are managed by pytorch_geometric
                    is_private = key.startswith("_")
                    if not is_private:
                        log.warning(
                            f"Adding attribute {key} to OFData without specifying the representation. The key {key} will not transform."
                        )
            setattr(self._store, key, value)

    def __setitem__(self, key: str, value: Any):
        """Overwrite the default ``__setitem__`` to check if fields are added without their
        transformation order."""
        if "representations" in self:
            if key not in self.representations.keys() and not key.startswith("_"):
                log.warning(f"Setting item {key} to OFData without specifying the representation.")
        self._store[key] = value
