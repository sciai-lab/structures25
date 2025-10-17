"""Energies class for storing calculated energies.

This module contains the Energies class, which is used for storing the calculated energies of a
molecule. The class gives access to the total energy, as well as the electronic energy and the sum
of all 'initialized' energies. Also contains a __str__ method for printing the energies in a nice
format.
"""
import numpy as np
from pyscf import gto
from rich.table import Table

from mldft.utils.rich_utils import format_table_rich


def _format_energy(energy: float) -> str:
    """Formats an energy."""
    return f"{energy:5.5f}"


def _format_energies(energies: list[float]) -> list[str]:
    """Formats a list of energies."""
    return [_format_energy(energy) for energy in energies]


class Energies:
    """Class for storing calculated energies.

    Attributes:
        mol: The molecule. This is used to calculate the nuclear repulsion energy.
        energies_dict: The stored energies.
    """

    energies_dict: dict[str, float]
    mol: gto.Mole | None

    def __init__(
        self,
        mol: gto.Mole | None = None,
        **energies_dict: float,
    ) -> None:
        """Initializes the Energies class.

        Args:
            mol: The molecule. This is used to calculate the nuclear repulsion energy.
            **energies_dict: The energies to be stored. The keys should be the names of the
                energies, the values the energies themselves.
        """
        self.energies_dict = energies_dict

        if mol is not None:
            if "nuclear_repulsion" in self.energies_dict:
                raise ValueError("Both mol and nuclear_repulsion given.")
            else:
                nuclear_repulsion = mol.energy_nuc()
                self["nuclear_repulsion"] = nuclear_repulsion

    def __getitem__(self, key: str) -> float:
        """Returns the energy corresponding to the given key."""
        return self.energies_dict[key]

    def __setitem__(self, key: str, value: float) -> None:
        """Sets the energy corresponding to the given key."""
        self.energies_dict[key] = value

    @property
    def total_energy(self) -> float:
        """Calculates the total energy: Electronic + nuclear repulsion."""
        return self.electronic_energy + self.energies_dict["nuclear_repulsion"]

    @property
    def electronic_energy(self) -> float:
        """Calculates the  sum of the electronic energies."""
        ret = 0.0

        if "tot" in self.energies_dict:
            ret = self.energies_dict["tot"] - self.energies_dict["nuclear_repulsion"]
        else:
            for key in self.energies_dict:
                if key != "nuclear_repulsion":
                    ret += self.energies_dict[key]

        return ret

    @property
    def sum(self) -> float:
        """Calculates the sum of all initialized energies."""
        return sum(self.energies_dict.values())

    def __str__(self) -> str:
        """Returns a string representation of the energies, formatted as a table."""

        energy_names = list(self.energies_dict.keys())
        energy_strings = [_format_energy(energy) for energy in self.energies_dict.values()]

        extra_energy_names = ["electronic_energy"]
        extra_energy_values = [self.electronic_energy]
        if "nuclear_repulsion" in self.energies_dict:
            extra_energy_names.append("total_energy")
            extra_energy_values.append(self.total_energy)
        extra_energy_strings = [_format_energy(energy) for energy in extra_energy_values]

        return format_table_rich(
            ("Contribution", energy_names, extra_energy_names),
            ("Energy [Ha]", energy_strings, extra_energy_strings),
            col_kwargs=[{}, {"justify": "right"}],
            as_string=True,
        )

    def comparison_table(
        self, other: "Energies", names=("self", "other"), as_string=True
    ) -> Table | str:
        """Returns a comparison table of the energies.

        Args:
            other: The other Energies object to compare to.
            names: The names of the two Energies objects. Default is ('self', 'other').
            as_string: Whether to return the table as a string. Default is True.
        """

        energy_names = list(set(self.energies_dict.keys()).union(other.energies_dict.keys()))
        energies_self = [self.energies_dict.get(name, np.nan) for name in energy_names]
        energies_other = [other.energies_dict.get(name, np.nan) for name in energy_names]
        energy_diffs = list(np.array(energies_self) - np.array(energies_other))

        extra_energy_names = ["electronic_energy", "total_energy"]
        extra_energies_self = [
            self.electronic_energy,
            self.total_energy if "nuclear_repulsion" in self.energies_dict else None,
        ]
        extra_energies_other = [
            other.electronic_energy,
            other.total_energy if "nuclear_repulsion" in other.energies_dict else None,
        ]
        extra_energy_diffs = list(np.array(extra_energies_self) - np.array(extra_energies_other))

        return format_table_rich(
            ("Contribution", energy_names, extra_energy_names),
            (names[0], _format_energies(energies_self), _format_energies(extra_energies_self)),
            (names[1], _format_energies(energies_other), _format_energies(extra_energies_other)),
            ("Difference", _format_energies(energy_diffs), _format_energies(extra_energy_diffs)),
            col_kwargs=[{}] + [{"justify": "right"}] * 3,
            as_string=as_string,
        )
