from collections.abc import Callable
from functools import partial

import numpy as np
import torch
from loguru import logger
from pyscf import dft
from torch_geometric.nn import radius_graph

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.of_data import OFData, Representation
from mldft.ml.models.components.loss_function import project_gradient
from mldft.ofdft.basis_integrals import (
    get_coulomb_matrix,
    get_nuclear_attraction_vector,
    get_overlap_matrix,
)
from mldft.utils.grids import grid_setup
from mldft.utils.molecules import build_molecule_ofdata
from mldft.utils.sparse import construct_block_diag_coo_indices_and_shape


def apply_to_attributes(f: Callable, sample: OFData, attributes: tuple) -> OFData:
    """Simplified version of :meth:`Data.apply`, with the crucial difference that errors are not
    quietly ignored.

    Applies the function f to the attributes of the object obj specified in the tuple attributes.

    .. warning::
        This function modifies the object in-place.

    Args:
        f: The function to apply.
        sample: The object to apply the function to.
        attributes: The names of the attributes to apply the function to.

    Returns:
        The object with the attributes modified.
    """
    for attr in attributes:
        try:
            setattr(sample, attr, f(getattr(sample, attr)))
        except Exception:
            raise RuntimeError(f"Error applying {f} to attribute {attr} of {sample}.")
    return sample


def split_by_atom(
    sample: OFData, fields=("coeffs", "ground_state_coeffs", "gradient_label")
) -> OFData:
    """Split all basis-function wise fields by atom. Currently, not planned to be used, but
    demonstrates how fields can be split when needed.

    Args:
        sample: The sample.
        fields: The fields to split.

    Returns:
        The sample with the fields split by atom.
    """
    for key in fields:
        setattr(sample, key, np.split(getattr(sample, key), sample.atom_ptr[:-1]))
    return sample


class SplitByAtom:
    """Split all basis-function wise fields by atom."""

    def __call__(self, sample: OFData) -> OFData:
        """Apply the transform to the sample.

        See :func:`split_by_atom` for details.
        """
        return split_by_atom(sample)


class ProjectGradient(torch.nn.Module):
    """Project gradients stored in a sample onto the tangent space of the manifold of constant
    electron number."""

    def forward(self, sample: OFData) -> OFData:
        """Project the fields of the sample as configured in  :meth:`__init__`.

        Args:
            sample (OFData): The sample.

        Returns:
            OFData: The sample with the projected gradients
        """
        for key, representation in sample.representations.items():
            if representation == Representation.GRADIENT and hasattr(sample, key):
                sample[key] = project_gradient(getattr(sample, key), sample)
        return sample


def str_to_torch_float_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Convert a string to a torch float dtype.

    Useful to set the dtype in hydra configs.
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "torch.float64":
        return torch.float64
    elif dtype == "torch.float32":
        return torch.float32
    else:
        raise ValueError(f"Unknown dtype {dtype}")


def dtype_map(dtype: np.dtype, float_dtype: np.dtype | None | str = None) -> None | torch.dtype:
    """Map a numpy dtype to a torch dtype.

    Args:
        dtype: The numpy dtype.
        float_dtype: The dtype to use for float dtypes. Defaults to the pytorch default dtype. For strings,
            "torch.float64" and "torch.float32" are supported.

    Returns:
        The torch dtype, or None if no mapping exists.
    """
    # This adds hydra support for torch.float64 and torch.float32
    if type(float_dtype) == str:
        float_dtype = str_to_torch_float_dtype(float_dtype)
    if (
        dtype == np.float64
        or dtype == np.float32
        or dtype == np.float16
        or dtype == torch.float64
        or dtype == torch.float32
        or dtype == torch.float16
        or dtype == torch.bfloat16
    ):
        return torch.get_default_dtype() if float_dtype is None else float_dtype
    elif dtype == np.uint8:
        return torch.int
    else:
        return None


def tensor_or_array_to_torch(
    x: torch.Tensor | np.ndarray,
    device=None,
    float_dtype: np.dtype | torch.dtype | str | None = None,
) -> torch.Tensor:
    dtype = dtype_map(x.dtype, float_dtype=float_dtype)
    return torch.as_tensor(x, dtype=dtype, device=device)


def to_torch(
    sample: OFData, device=None, float_dtype: np.dtype | torch.dtype | str | None = None
) -> OFData:
    """Convert all numpy arrays in the sample to torch tensors.

    Args:
        sample: The sample.
        device: The device to put the tensors on. Defaults to None, i.e. the pytorch default device.
        float_dtype: The dtype to use for float dtypes. Defaults to the pytorch default dtype. For strings, "torch.float64"
            and "torch.float32" are supported to enable hydra support.

    Returns:
        The sample with all numpy arrays converted to torch tensors.
    """
    keys = []
    for key in sample.keys():
        if isinstance(getattr(sample, key), np.ndarray):
            if not getattr(sample, key).dtype == np.object_:
                keys.append(key)
        elif isinstance(getattr(sample, key), torch.Tensor):
            keys.append(key)
    func = partial(tensor_or_array_to_torch, device=device, float_dtype=float_dtype)
    apply_to_attributes(func, sample, keys)
    return sample


class ToTorch:
    """Convert all numpy arrays in the sample to torch tensors."""

    def __init__(
        self,
        device: torch.device = None,
        float_dtype: np.dtype | torch.dtype | None = None,
    ):
        """Initialize the transform.

        Args:
            device: The device to put the tensors on. Defaults to None, i.e. the pytorch default device.
            float_dtype: The dtype to use for float dtypes. Defaults to the pytorch default dtype. For strings, "torch.float64"
                and "torch.float32" are supported.
        """
        self.device = device
        self.float_dtype = float_dtype

    def __call__(self, sample: OFData) -> OFData:
        """Apply the transform to the sample.

        See :func:`to_torch` for details.
        """
        return to_torch(sample, device=self.device, float_dtype=self.float_dtype)


def to_numpy(sample: OFData) -> OFData:
    """Convert all torch tensors in the sample to numpy arrays."""
    keys = tuple(key for key in sample.keys() if isinstance(getattr(sample, key), torch.Tensor))
    return apply_to_attributes(lambda x: x.detach().cpu().numpy(), sample, keys)


class ToNumpy:
    """Convert all torch tensors in the sample to numpy arrays."""

    def __call__(self, sample: OFData) -> OFData:
        """Apply the transform to the sample.

        See :func:`to_numpy` for details.
        """
        return to_numpy(sample)


def add_atom_coo_indices(sample: OFData) -> OFData:
    """Add a field "atom_coo_indices" to the sample, containing an index tensor of shape (2,
    n_basis), which can be used to construct a block diagonal sparse matrix with each block
    corresponding to the basis functions of one atom.

    Args:
        sample: The sample.

    Returns:
        The sample with the field "atom_coo_indices" added.
    """
    block_shapes = [(n_basis, n_basis) for n_basis in sample.n_basis_per_atom]
    sample.add_item(
        "atom_coo_indices",
        construct_block_diag_coo_indices_and_shape(*block_shapes)[0],
        representation=Representation.NONE,
    )
    return sample


class AddAtomCooIndices:
    """Add a field "atom_coo_indices" to the sample, containing an index tensor of shape (2,
    n_basis), which can be used to construct a block diagonal sparse matrix with each block
    corresponding to the basis functions of one atom."""

    def __call__(self, sample: OFData) -> OFData:
        """Apply the transform to the sample.

        See :func:`add_atom_coo_indices` for details.
        """
        return add_atom_coo_indices(sample)


class AddOverlapMatrix:
    """Adds the overlap matrix to the sample."""

    def __init__(self, basis_info: BasisInfo):
        """
        Args:
            basis_info: The basis information in the BasisInfo format.
        """
        super().__init__()
        self.basis_info = basis_info

    def __call__(self, sample: OFData) -> OFData:
        """
        Args:
            sample: the molecule in the OFData format
        """
        mol = build_molecule_ofdata(sample, self.basis_info.basis_dict)
        # The overlap matrix is only ever needed to compute the natrep transformation matrix.
        sample.add_item("overlap_matrix", get_overlap_matrix(mol), Representation.BILINEAR_FORM)
        return sample


class AddFullEdgeIndex(torch.nn.Module):
    """Add a full edge index to the sample."""

    def forward(self, sample: OFData) -> OFData:
        """Add a full edge index to the sample and returns the same sample with the additional
        edge_index attribute.

        Args:
            sample (OFData): Sample data object.

        Returns:
            OFData: The same sample data object with the additional edge_index attribute.
        """

        number_nodes = sample.pos.shape[0]
        edge_index = torch.stack(
            torch.meshgrid(torch.arange(number_nodes), torch.arange(number_nodes), indexing="ij")
        ).reshape(2, -1)

        sample.add_item("edge_index", edge_index, Representation.NONE)
        return sample


class AddRadiusEdgeIndex(torch.nn.Module):
    """Add a radius edge index to the sample."""

    def __init__(self, radius: float):
        """
        Args:
            radius: The radius to use for the edge index.
        """
        super().__init__()
        self.radius = radius

    def forward(self, sample: OFData) -> OFData:
        """Add a radius edge index to the sample and returns the same sample with the additional
        edge_index attribute.

        Args:
            sample (OFData): Sample data object.

        Returns:
            OFData: The same sample data object with the additional edge_index attribute.
        """

        pos = sample.pos
        edge_index = radius_graph(pos, r=self.radius, loop=True)
        sample.add_item("edge_index", edge_index, Representation.NONE)
        return sample


class AddBasisInfo:
    """Adds the nuclear attraction vector to the sample."""

    def __init__(self, basis_info: BasisInfo):
        """
        Args:
            basis_info: The basis information in the BasisInfo format.
        """
        super().__init__()
        self.basis_info = basis_info

    def __call__(self, sample: OFData) -> OFData:
        """
        Args:
            sample: the molecule in the OFData format
        """

        sample.add_item("basis_info", self.basis_info, Representation.NONE)
        return sample


class AddNuclearAttractionVector:
    """Adds the nuclear attraction vector to the sample."""

    def __init__(self, basis_info: BasisInfo):
        """
        Args:
            basis_info: The basis information in the BasisInfo format.
        """
        super().__init__()
        self.basis_info = basis_info

    def __call__(self, sample: OFData) -> OFData:
        """
        Args:
            sample: the molecule in the OFData format
        """
        mol = build_molecule_ofdata(sample, self.basis_info.basis_dict)
        nuclear_attraction_vector = torch.as_tensor(
            get_nuclear_attraction_vector(mol), dtype=torch.float64
        )

        sample.add_item(
            "nuclear_attraction_vector",
            nuclear_attraction_vector,
            Representation.DUAL_VECTOR,
        )
        return sample


class AddCoulombMatrix:
    """Adds the coulomb matrix to the sample."""

    def __init__(self, basis_info: BasisInfo):
        """
        Args:
            basis_info: The basis information in the BasisInfo format.
        """
        super().__init__()
        self.basis_info = basis_info

    def __call__(self, sample: OFData) -> OFData:
        """
        Args:
            sample: the molecule in the OFData format
        """
        mol = build_molecule_ofdata(sample, self.basis_info.basis_dict)
        coulomb_matrix = torch.as_tensor(get_coulomb_matrix(mol), dtype=torch.float64)

        sample.add_item("coulomb_matrix", coulomb_matrix, Representation.BILINEAR_FORM)
        return sample


class AddMol:
    """Adds the mol to the sample."""

    def __init__(self, basis_info: BasisInfo):
        """
        Args:
            basis_info: The basis information in the BasisInfo format.
        """
        super().__init__()
        self.basis_info = basis_info

    def __call__(self, sample: OFData) -> OFData:
        """
        Args:
            sample: the molecule in the OFData format
        """
        mol = build_molecule_ofdata(sample, self.basis_info.basis_dict)

        sample.add_item("mol", mol, Representation.NONE)
        return sample


class PrepareForDensityOptimization:
    def __init__(
        self,
        basis_info: BasisInfo,
        add_grid: bool = False,
        grid_level: int = 2,
        grid_prune: str = "nwchem_prune",
    ):
        """Transform adding the necessary information for density optimization."""
        self.basis_info = basis_info
        self.add_grid = add_grid
        self.grid_level = grid_level
        self.grid_prune = grid_prune

    def __call__(self, sample: OFData) -> OFData:
        """Prepare the sample for density optimization.

        Args:
            sample: The sample.

        Returns:
            OFData: The prepared sample.
        """
        mol = build_molecule_ofdata(sample, self.basis_info.basis_dict)
        sample.add_item("mol", mol, representation=Representation.NONE)
        if not hasattr(sample, "basis_info"):
            sample.add_item("basis_info", self.basis_info, representation=Representation.NONE)
        if not hasattr(sample, "overlap_matrix"):
            sample = AddOverlapMatrix(basis_info=self.basis_info)(sample)
        if not hasattr(sample, "coulomb_matrix"):
            coulomb_matrix = get_coulomb_matrix(mol)
            sample.add_item(
                "coulomb_matrix",
                coulomb_matrix,
                representation=Representation.BILINEAR_FORM,
            )
        if not hasattr(sample, "nuclear_attraction_vector"):
            nuclear_attraction_vector = get_nuclear_attraction_vector(mol)
            sample.add_item(
                "nuclear_attraction_vector",
                nuclear_attraction_vector,
                representation=Representation.DUAL_VECTOR,
            )
        if self.add_grid:
            logger.info("Adding grid to sample.")
            grid = grid_setup(mol, self.grid_level, self.grid_prune)
            sample.add_item("grid_level", self.grid_level, representation=Representation.NONE)
            sample.add_item("grid_prune", self.grid_prune, representation=Representation.NONE)
            ao = np.asarray(dft.numint.eval_ao(mol, grid.coords, deriv=1))
            sample.add_item("ao", ao, representation=Representation.AO)
        return sample
