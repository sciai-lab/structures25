import tempfile

import numpy as np
import pyscf
import pytest

from mldft.utils.molecules import (  # replace 'your_module' with the module of your function
    read_xyz_file,
)

test_files = [
    (
        "3\n"
        "comment line\n"
        "H 0.0 0.0 0.0\n"  # Hydrogen atom at origin
        "H 0.0 0.0 1.0\n"  # Hydrogen atom at (0, 0, 1)
        "O 0.0 1.0 0.0\n"
    ),  # Oxygen atom at (0, 1, 0)
    (
        "4\n"
        "comment line\n"
        "H 0.0 0.0 0.0\n"  # Hydrogen atom at origin
        "H 0.0 0.0 1.0\n"  # Hydrogen atom at (0, 0, 1)
        "O 0.0 1.0 0.0\n"  # Oxygen atom at (0, 1, 0)
        "C 1.0 1.0 1.0\n"
    ),  # Carbon atom at (1, 1, 1)
    (
        "5\n"
        "comment line\n"
        "H 0.0 0.0 0.0\n"  # Hydrogen atom at origin
        "H 0.0 0.0 1.0\n"  # Hydrogen atom at (0, 0, 1)
        "O 0.0 1.0 0.0\n"  # Oxygen atom at (0, 1, 0)
        "C 1.0 1.0 1.0\n"  # Carbon atom at (1, 1, 1)
        "S 1.414 1.414 1.414\n"
    ),  # Nitrogen atom at (1.414, 1.414, 1.414)
    (
        "6\n"
        "comment line\n"
        "H 0.0 0.0 0.0\n"  # Hydrogen atom at origin
        "H 1.0 0.0 0.0\n"  # Hydrogen atom at (1, 0, 0)
        "O 0.0 1.0 0.0\n"  # Oxygen atom at (0, 1, 0)
        "C 0.0 0.0 1.0\n"  # Carbon atom at (0, 0, 1)
        "N 1.4651 1.354152 1.98135\n"  # Nitrogen atom at (1.4651, 1.354152, 1.98135)
        "F 2.35543 2.53145 2.641648\n"
    ),  # Sulfur atom at (2.35543, 2.53145, 2.641648)
]


@pytest.mark.parametrize("test_file_content", test_files)
def test_read_xyz_file(test_file_content):
    """Test the read_xyz_file function by comparing with pyscf."""
    # Create a temporary test .xyz file
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=True) as temp:
        test_file_path = temp.name
        with open(test_file_path, "w") as f:
            f.write(test_file_content)

        # Read the file with the function to test
        atomic_numbers, coordinates = read_xyz_file(test_file_path)

        # Read the file with pyscf
        mol = pyscf.gto.M(atom=test_file_path, basis="ccpvdz", unit="Angstrom")
        pyscf_atomic_numbers = mol.atom_charges()
        pyscf_coordinates = mol.atom_coords(unit="Angstrom")

        # Check if atomic numbers match
        assert np.array_equal(
            atomic_numbers, pyscf_atomic_numbers
        ), "Atomic numbers do not match between read_xyz_file and pyscf"

        # Check if coordinates match
        assert np.allclose(
            coordinates, pyscf_coordinates
        ), "Coordinates do not match between read_xyz_file and pyscf"
