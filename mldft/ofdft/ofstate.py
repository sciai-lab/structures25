"""Contains the class for storing the state of the OFDFT calculation."""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from pyscf import dft, gto

from mldft.ofdft.energies import Energies
from mldft.utils.pyscf_pretty_print import mol_to_str


class StoppingCriterion(Enum):
    """Enum for the stopping criterion.

    The density optimization is run over a fixed number of steps. The stopping criterion
    determines which result is chosen as the final result.

    Attributes:
        GRADIENT_STOPS_DECREASING: The projected gradient stopped decreasing (first local
            minimum of the projected gradient unless constant values are reached).
        ENERGY_UPDATE_STOPS_DECREASING: The energy update stopped decreasing (first local minimum
            of a single-step energy unless constant values are reached).
        ENERGY_UPDATE_GLOBAL_MINIMUM: The minimal single-step energy update.
    """

    GRADIENT_STOPS_DECREASING = "projected gradient stopped decreasing"
    ENERGY_UPDATE_STOPS_DECREASING = "energy update stopped decreasing"
    ENERGY_UPDATE_GLOBAL_MINIMUM = "energy update global minimum"
    LAST_ITERATION = "last iteration"


@dataclass
class OFState:
    """Class for storing the state of the OFDFT calculation.

    Attributes:
        mol: The molecule.
        coeffs: The coefficients of the electron density.
        grid: The grid used for the calculation.
        energy: The calculated energies.
    """

    mol: gto.Mole
    coeffs: np.ndarray | None = None
    grid: dft.Grids | None = None
    energy: Energies | None = None
    stopping_criterion: StoppingCriterion | None = None
    stopping_index: int | None = None

    def __str__(self) -> str:
        """Returns a string representation of the state."""
        string = f"Molecule: {mol_to_str(self.mol)}"

        if self.coeffs is not None:
            string += f"\nCoeffs: {str(self.coeffs)}"

        if self.grid is not None:
            string += f"\nGrid level: {self.grid.level}"
            string += f"\nGrid prune: {self.grid.prune}"
            string += f"\nGrid size: {self.grid.size}"

        if self.energy is not None:
            string += f"\nEnergies:\n{self.energy.__str__()}"

        return string
