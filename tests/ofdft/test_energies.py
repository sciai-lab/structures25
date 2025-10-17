import numpy as np
import pytest
from pyscf import gto

from mldft.ofdft.energies import Energies


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mol": gto.M(atom="H 0 0 0; F 0 0 1")},
        {
            "mol": gto.M(atom="H 0 0 0; F 0 0 1"),
            "kinetic": 1.0,
            "hartree": 2.0,
            "xc": 3.0,
            "nuclear_attraction": 4.0,
        },
        {"kinetic": 1.0, "hartree": 2.0, "xc": 3.0, "nuclear_attraction": 4.0},
        {"kinetic": 1.0, "xc": 3.0},
    ],
)
def test_Energies__init__(kwargs):
    """Checks whether the Energies class is initialized correctly."""
    energies = Energies(**kwargs)
    for key, value in kwargs.items():
        if key == "mol" and value is not None:
            value = kwargs[key].energy_nuc()
            key = "nuclear_repulsion"
        else:
            pass
        assert np.isclose(energies[key], value, equal_nan=True), f"Failed for {key}."


@pytest.mark.parametrize(
    "energies,expected_sum",
    [
        (Energies(kinetic=1.0, hartree=2.0, xc=3.0, nuclear_attraction=4.0), 10.0),
        (Energies(kinetic=1.0, hartree=2.0, xc=3.0), 6.0),
        (Energies(kinetic=1.0, xc=3.0), 4.0),
        (Energies(kinetic=1.0), 1.0),
        (Energies(), 0.0),
    ],
)
def test_Energies_sum(energies, expected_sum):
    """Checks whether the sum energy is calculated correctly."""
    assert np.isclose(energies.sum, expected_sum, equal_nan=False), f"Failed for {energies}."


@pytest.mark.parametrize(
    "energies,expected_sum",
    [
        (Energies(kinetic=1.0, hartree=2.0, xc=3.0, nuclear_attraction=4.0), 10.0),
        (Energies(kinetic=1.0, hartree=2.0, xc=3.0), 6.0),
    ],
)
def test_Energies_eletronic_energy(energies, expected_sum):
    """Checks whether the electronic energy is calculated correctly."""
    assert np.isclose(
        energies.electronic_energy, expected_sum, equal_nan=True
    ), f"Failed for {energies}."


@pytest.mark.parametrize(
    "energies, energies_sum",
    [
        (
            Energies(
                kinetic=1.0,
                hartree=2.0,
                xc=3.0,
                nuclear_attraction=4.0,
                mol=gto.M(atom="H 0 0 0; F 0 0 1"),
            ),
            10.0 + gto.M(atom="H 0 0 0; F 0 0 1").energy_nuc(),
        ),
    ],
)
def test_Energies_total_energy(energies, energies_sum):
    """Checks whether the total energy is calculated correctly."""
    assert np.isclose(
        energies.total_energy, energies_sum, equal_nan=True
    ), f"Failed for {energies}."


def test_Energies__str__():
    """Checks whether the __str__ method is working."""
    test_mol = gto.M(atom="H 0 0 0; F 0 0 1", basis="cc-pVDZ")
    test_energies = Energies(kinetic=1, hartree=2, xc=3, nuclear_attraction=4, mol=test_mol)
    print(str(test_energies))
    expected_str = (
        "┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓\n"
        "┃ Contribution       ┃ Energy [Ha] ┃\n"
        "┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩\n"
        "│ kinetic            │     1.00000 │\n"
        "│ hartree            │     2.00000 │\n"
        "│ xc                 │     3.00000 │\n"
        "│ nuclear_attraction │     4.00000 │\n"
        "│ nuclear_repulsion  │     4.76259 │\n"
        "├────────────────────┼─────────────┤\n"
        "│ electronic_energy  │    10.00000 │\n"
        "│ total_energy       │    14.76259 │\n"
        "└────────────────────┴─────────────┘\n"
    )
    assert str(test_energies) == expected_str
