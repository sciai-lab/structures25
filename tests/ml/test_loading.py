import numpy as np
import pytest
import torch
from e3nn.o3 import Irreps
from torch_geometric.data import Batch, Data

from mldft.ml.data.components.basis_transforms import ToLocalFrames
from mldft.ml.data.components.convert_transforms import (
    AddAtomCooIndices,
    ToTorch,
    add_atom_coo_indices,
    split_by_atom,
    to_numpy,
    to_torch,
)
from mldft.ml.data.components.dataset import OFDataset
from mldft.ml.data.components.loader import OFLoader
from mldft.ml.data.components.of_data import OFData


def check_sample_or_batch(sample_or_batch: Data | Batch, n_atom=None) -> None:
    """Checking the presence, shapes and consistency of some of the sample's (or batch's)
    attributes."""
    n_atom = len(sample_or_batch.atomic_numbers) if n_atom is None else n_atom
    n_scalars = len(sample_or_batch) if isinstance(sample_or_batch, Batch) else 1
    assert sample_or_batch.pos.shape == (n_atom, 3)
    assert sample_or_batch.atomic_numbers.shape == (n_atom,)
    assert sample_or_batch.atom_ind.shape == (n_atom,)
    assert sample_or_batch.n_basis_per_atom.shape == (n_atom,)
    assert sample_or_batch.atom_ptr.shape == (n_atom,)
    assert sample_or_batch.energy_label.shape == (n_scalars,)

    # check that n_basis_per_atom and atom_ptr are consistent
    coeffs_numpy = (
        sample_or_batch.coeffs.numpy()
        if isinstance(sample_or_batch.coeffs, torch.Tensor)
        else sample_or_batch.coeffs
    )
    split_by_n_basis = torch.split(
        torch.from_numpy(coeffs_numpy), sample_or_batch.n_basis_per_atom.tolist()
    )
    split_py_atom_ptr = np.split(coeffs_numpy, sample_or_batch.atom_ptr[:-1])
    assert len(split_by_n_basis) == len(split_py_atom_ptr)
    assert np.allclose([x.shape for x in split_by_n_basis], [x.shape for x in split_py_atom_ptr])
    assert np.allclose(split_by_n_basis[0], split_py_atom_ptr[0])
    assert np.allclose(split_by_n_basis[-1], split_py_atom_ptr[-1])

    n_basis = sample_or_batch.n_basis
    assert sample_or_batch.n_basis_per_atom.sum() == n_basis
    assert sample_or_batch.coeffs.shape == (n_basis,)
    assert sample_or_batch.ground_state_coeffs.shape == (n_basis,)
    assert sample_or_batch.gradient_label.shape == (n_basis,)
    assert sample_or_batch.dual_basis_integrals.shape == (n_basis,)

    # currently, the irreps are a numpy array of irreps per atom for a Data object,
    # and these numpy arrays for different samples (molecules) are collected in a list for a Batch object
    # Nicer would be to concatenate the numpy arrays, but I see no elegant way to do this with pytorch geometric.
    # (One option is a custom Loader AND overwriting Batch.to_data_list, but that seems like overkill.)

    if isinstance(sample_or_batch, Batch):
        assert isinstance(sample_or_batch.mol_id, list)
        assert len(sample_or_batch.mol_id) == len(sample_or_batch)

        assert isinstance(sample_or_batch.irreps_per_atom, list)
        assert (
            np.sum(
                [len(irreps_for_atom_i) for irreps_for_atom_i in sample_or_batch.irreps_per_atom]
            )
            == n_atom
        )
        for irreps_for_atom_i in sample_or_batch.irreps_per_atom:
            assert isinstance(irreps_for_atom_i, np.ndarray)
            assert isinstance(irreps_for_atom_i[0], Irreps)
    else:
        assert isinstance(sample_or_batch.mol_id, str)
        assert isinstance(sample_or_batch.irreps_per_atom, np.ndarray)
        assert len(sample_or_batch.irreps_per_atom) == n_atom
        assert isinstance(sample_or_batch.irreps_per_atom[0], Irreps)
        assert np.sum([irr.dim for irr in sample_or_batch.irreps_per_atom]) == n_basis

    # test splitting by atom
    coeffs_by_atom = sample_or_batch.split_field_by_atom(sample_or_batch.coeffs)
    assert len(coeffs_by_atom) == n_atom
    assert np.allclose([len(c) for c in coeffs_by_atom], sample_or_batch.n_basis_per_atom)


def test_load_sample(dummy_sample_path, dummy_basis_info):
    """Test loading a sample from a .zarr file.

    This is a very basic test, just checking the presence, shapes and some consistency of the
    attributes.
    """
    sample = OFData.from_file(dummy_sample_path, 0, dummy_basis_info, add_irreps=True)
    check_sample_or_batch(sample, n_atom=30)


def test_to_torch(dummy_sample):
    """Test the to_torch transform."""
    sample = to_torch(dummy_sample)
    assert isinstance(sample.pos, torch.Tensor)
    assert isinstance(sample.atomic_numbers, torch.Tensor)
    assert isinstance(sample.atom_ind, torch.Tensor)
    assert isinstance(sample.n_basis_per_atom, torch.Tensor)
    assert isinstance(sample.atom_ptr, torch.Tensor)
    assert isinstance(sample.coeffs, torch.Tensor)
    assert isinstance(sample.ground_state_coeffs, torch.Tensor)
    assert isinstance(sample.gradient_label, torch.Tensor)
    assert isinstance(sample.dual_basis_integrals, torch.Tensor)


def test_to_numpy(dummy_sample):
    """Test that the to_numpy transform indeed converts torch tensors to numpy arrays."""
    sample = to_numpy(to_torch(dummy_sample))
    assert isinstance(sample.pos, np.ndarray)
    assert isinstance(sample.atomic_numbers, np.ndarray)
    assert isinstance(sample.atom_ind, np.ndarray)
    assert isinstance(sample.n_basis_per_atom, np.ndarray)
    assert isinstance(sample.atom_ptr, np.ndarray)
    assert isinstance(sample.coeffs, np.ndarray)
    assert isinstance(sample.ground_state_coeffs, np.ndarray)
    assert isinstance(sample.gradient_label, np.ndarray)
    assert isinstance(sample.dual_basis_integrals, np.ndarray)


def test_add_atom_coo_indices(dummy_sample):
    """Test the add_atom_coo_indices transform:
    make sure that the atom_coo_indices attribute is added and the shape is as expected"""
    sample = add_atom_coo_indices(dummy_sample.clone())
    assert sample.atom_coo_indices.shape == (
        2,
        np.sum([n_basis**2 for n_basis in sample.n_basis_per_atom]),
    )


def test_split_by_atom(dummy_sample):
    """Test the split_by_atom transform."""
    sample = split_by_atom(dummy_sample, fields=[])
    check_sample_or_batch(sample)

    sample = split_by_atom(dummy_sample, fields=["coeffs"])
    assert len(sample.coeffs) == sample.num_nodes
    assert np.allclose(sample.n_basis_per_atom, [len(c) for c in sample.coeffs])


@pytest.mark.parametrize("sparse", [False, True])
def test_to_local_frames(dummy_sample, sparse):
    """Test the to_local_frames transform."""
    dummy_sample_torch = ToTorch()(dummy_sample)
    if sparse:
        dummy_sample_torch = AddAtomCooIndices()(dummy_sample_torch)
    sample = ToLocalFrames(sparse=sparse)(dummy_sample_torch)
    check_sample_or_batch(sample)


def test_dataset(dummy_dataset_path, dummy_basis_info):
    """Test loading a dataset from a directory containing .zarr files.

    This is a very basic test, just checking the presence, shapes and some consistency of the
    attributes.
    """
    dataset = OFDataset.from_directory(
        dummy_dataset_path, basis_info=dummy_basis_info, add_irreps=True
    )
    assert len(dataset) == np.sum(np.arange(1, 11))
    for sample in dataset:
        check_sample_or_batch(sample)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_loader(dummy_dataset_torch, batch_size):
    """Test the loader."""
    loader = OFLoader(dummy_dataset_torch, batch_size=batch_size, shuffle=True, drop_last=True)
    for batch in loader:
        assert len(batch) == batch_size
        check_sample_or_batch(batch)
        for sample in batch.to_data_list():
            check_sample_or_batch(sample)
