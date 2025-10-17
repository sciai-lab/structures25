import numpy as np
import pytest

from mldft.utils.molecules import build_molecule_np

# List of tuples of (charges, positions) for testing.
test_atoms = [
    (np.array([8, 1, 1]), np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])),
    (
        np.array([6, 6, 8, 1, 1, 1, 1, 1, 1]),
        np.array(
            [
                [0.0072, -0.5687, 0.0],
                [-1.2854, 0.2499, 0.0],
                [1.1304, 0.3147, 0.0],
                [0.0392, -1.1972, 0.89],
                [0.0392, -1.1972, -0.89],
                [-1.3175, 0.8784, 0.89],
                [-1.3175, 0.8784, -0.89],
                [-2.1422, -0.4239, 0.0],
                [1.9857, -0.1365, 0.0],
            ]
        ),
    ),
]


@pytest.mark.parametrize("charges, positions", test_atoms)
def test_build_molecule_np(charges: np.ndarray, positions: np.ndarray):
    """Test the build_molecule_np function."""
    molecule = build_molecule_np(charges, positions, basis="6-31G(2df,p)", unit="Angstrom")
    assert molecule.unit == "Angstrom"
    assert molecule.charge == 0
    assert np.all(molecule.atom_charges() == charges)
    assert np.allclose(molecule.atom_coords(unit="Angstrom"), positions)
