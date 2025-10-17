import numpy as np
import pytest
import torch

from mldft.ml.data.components.convert_transforms import to_torch
from mldft.ml.data.components.of_data import OFData
from mldft.ml.models.components.atom_ref import AtomRef


@pytest.fixture
def atom_ref_setup(dummy_basis_info):
    """Setup for the atom ref fitter tests.

    Returns:
        data: The data for the tests.
        n_atomic_numbers: The number of atom types.
        atomic_number_to_irreps: The atom type to irreps mapping.
    """

    sample = OFData.construct_new(
        basis_info=dummy_basis_info,
        pos=np.zeros((3, 3), dtype=np.float32),  # not needed for AtomRef
        atomic_numbers=np.array([1, 8, 1], dtype=np.int32),  # H2O molecule
        coeffs=np.concatenate(
            [
                np.array([1] + [0] * 5),  # H
                np.array([1] + [0] * 17),  # O
                np.array([1] + [0] * 5),  # H
            ]
        ).astype(np.float32),
        gradient_label=np.concatenate(
            [
                np.array([1] + [0] * 5),  # H
                np.array([1] + [0] * 17),  # O
                np.array([1] + [0] * 5),  # H
            ],
        ).astype(np.float32),
        energy_label=np.array([28], dtype=np.float32),
        dual_basis_integrals="infer_from_basis",
    )
    sample = to_torch(sample)

    g_z = np.concatenate(
        [
            np.array([1] + [0] * 5),  # H
            np.array([1] + [0] * 17),  # C
            np.array([2] + [0] * 17),  # O
        ]
    ).astype(np.float32)
    t_z = torch.FloatTensor([3, 1, 4])  # H, C, O
    t_global = 1
    n_atom_types = 2

    return sample, n_atom_types, g_z, t_z, t_global


def test_atom_ref(atom_ref_setup):
    """Test the atomic reference module for an example molecule.

    Args:
        atom_ref_setup: Setup for the tests for the atomic reference module
    """
    sample, n_atom_types, g_z, t_z, t_global = atom_ref_setup
    atom_ref = AtomRef(t_z=t_z, t_global=t_global, g_z=g_z)

    reference_energy = atom_ref.sample_forward(sample)

    assert reference_energy == pytest.approx((1 + 1 + 2) + (3 + 3 + 4) + 1)


def test_atom_ref_from_statistics(dummy_dataset_statistics, dummy_loader):
    """Test the atomic reference module for batches from the dummy loader."""

    atom_ref = AtomRef.from_dataset_statistics(dummy_dataset_statistics, weigher_key="constant")

    for batch in dummy_loader:
        atom_ref_energy = atom_ref.sample_forward(batch)
        assert atom_ref_energy.shape == (batch.num_graphs,)
