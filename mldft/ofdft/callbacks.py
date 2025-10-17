"""Callback classes for the minimization."""

import dataclasses
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from pyscf import gto

from mldft.ml.data.components.of_data import StrEnumwithCheck
from mldft.ofdft.energies import Energies
from mldft.ofdft.ofstate import OFState, StoppingCriterion


@dataclasses.dataclass
class BasicCallback:
    """Callback class for basic information during the minimization.

    The basic callback is used to store the energy, gradient norm, learning rate and the
    coefficients during the minimization. It can be used to plot the minimization
    progress.

    Attributes:
        mol (Optional[gto.Mole]): PySCF Mole object
        energy (List[Energies]): List of electronic energies
        gradient_norm (List[float]): List of gradient norm
        learning_rate (List[float]): List of learning rates
        coeffs (List[np.ndarray]): List of coefficients
    """

    mol: Optional[gto.Mole] = None

    energy: List[Energies] = dataclasses.field(default_factory=list)
    gradient_norm: List[float] = dataclasses.field(default_factory=list)
    l2_norm: List[float] = dataclasses.field(default_factory=list)
    learning_rate: List[float] = dataclasses.field(default_factory=list)
    coeffs: List[torch.Tensor] = dataclasses.field(default_factory=list)
    stopping_index: Optional[int] = None

    def __call__(self, minimization_locals: dict) -> None:
        """Callback function for the minimization.

        Args:
            minimization_locals (dict): Dictionary containing the local variables of the
                minimization function.
        """
        self.energy.append(minimization_locals["energy"])
        self.gradient_norm.append(minimization_locals["gradient_norm"])
        self.learning_rate.append(minimization_locals["learning_rate"])
        coeffs = minimization_locals["coeffs"]
        sample = minimization_locals["sample"]
        delta_coeffs = coeffs.detach() - sample.ground_state_coeffs
        l2_norm = torch.sqrt(delta_coeffs @ sample.overlap_matrix @ delta_coeffs).item()
        self.l2_norm.append(l2_norm)
        self.coeffs.append(coeffs.detach().cpu().clone())

    def convert_to_numpy(self) -> dict:
        """Convert the data to a numpy array.

        Returns:
            dict: Dictionary containing the data as numpy arrays.
        """
        data = dict()
        for field in dataclasses.fields(self):
            data[field.name] = np.asarray(getattr(self, field.name))

        return data

    def save_to_file(self, filename: str | Path) -> None:
        """Save the data to a npz file.

        Args:
            filename (str | Path): Filename.
        """
        data = self.convert_to_numpy()
        np.savez(filename, **data)

    @classmethod
    def from_npz(cls, filename: str | Path, mol: gto.Mole | None = None) -> "BasicCallback":
        """Load the data from a npz file.

        Args:
            filename (str | Path): Filename.
            mol (Optional[gto.Mole]): PySCF Mole object.

        Returns:
            BasicCallback: Callback object. If mol is not None, the mol attribute is
                set to mol.
        """
        callback = cls(mol=mol)
        data = np.load(filename, allow_pickle=True)
        for field in dataclasses.fields(callback):
            setattr(callback, field.name, data[field.name])
        return callback


class ConvergenceCriterion(StrEnumwithCheck):
    LAST_ITER = "last_iter"
    MIN_E_STEP = "min_e_step"
    MOFDFT = "mofdft"


class ConvergenceCallback(BasicCallback):
    """Callback that extends BasisCallback by the stopping criterion."""

    def get_convergence_result(
        self,
        convergence_criterion: str | ConvergenceCriterion = "last_iter",
    ) -> OFState:
        """Return the final "converged" result.

        If the criterion is last iteration, the last step is used.
        If the criterion is mof_small_scale, the final result is the one with the
        smallest energy difference. If the criterion is mof_large_scale, the final
        result is chosen hierarchically:

            #. The step where the projected gradient first stops decreasing, if existent.
               Unless there are constant values, this is the first local minimum of the
               projected gradient.
            #. The first step where the single-step energy update first stops
               decreasing, if existent. Unless there are constant values, this is the
               first local minimum of the single-step energy update.
            #. The minimal energy update (always exists).

        The energy update is to be understood as the absolute value of the difference.

        Args:
            convergence_criterion: The criterion to choose the final result.

        Returns:
            OFState of the chosen final result. It has the new attribute stopping_index and
            stopping_criterion, which contain the index of the chosen result and the
            :py:class:`~mldft.ofdft.ofstate.StoppingCriterion` that was used.

        Raises:
            ValueError: If the lengths of energy, gradient_norm and coeffs do not match.
        """
        ConvergenceCriterion.check_key(
            convergence_criterion
        ), f"Invalid convergence criterion: {convergence_criterion}"
        if convergence_criterion == "last_iter":
            state = OFState(
                mol=self.mol,
                coeffs=self.coeffs[-1],
                energy=self.energy[-1],
            )
            stopping_index = len(self.energy) - 1
            state.stopping_index = stopping_index
            state.stopping_criterion = StoppingCriterion.LAST_ITERATION
            self.stopping_index = stopping_index
            return state

        if not len(self.energy) == len(self.gradient_norm) == len(self.coeffs):
            raise ValueError("Length of energy, gradient_norm and coeffs must be the same.")
        if len(self.energy) == 0:
            raise ValueError("No data available.")
        elif len(self.energy) == 1:
            state = OFState(
                mol=self.mol,
                coeffs=self.coeffs[0],
                energy=self.energy[0],
            )
            state.stopping_index = 0
            state.stopping_criterion = StoppingCriterion.ENERGY_UPDATE_GLOBAL_MINIMUM
            return state

        energies = np.asarray([e.total_energy for e in self.energy])
        energy_diff = np.diff(energies)
        energy_updates = np.abs(energy_diff)

        # as a baseline, choose the smallest energy difference
        # since diff[i] = energy[i+1] - energy[i], the index is before the
        # smallest energy difference, so we are closer to the actual minimum
        stopping_index = np.argmin(energy_updates)
        stopping_criterion = StoppingCriterion.ENERGY_UPDATE_GLOBAL_MINIMUM

        if convergence_criterion == ConvergenceCriterion.MOFDFT:
            gradient_norms = np.asarray(self.gradient_norm)
            gradient_norm_diff = np.diff(gradient_norms)
            gradient_decreasing = gradient_norm_diff < 0

            energy_update_decreasing = np.diff(energy_updates) < 0

            if not np.all(gradient_decreasing):
                # choose index of the first local minimum of the gradient norm,
                # i.e. the first index where gradient_decreasing is False
                stopping_index = np.argmin(gradient_decreasing)
                stopping_criterion = StoppingCriterion.GRADIENT_STOPS_DECREASING

            elif not np.all(energy_update_decreasing):
                # choose index of the first local minimum of the energy difference,
                # i.e. the first index where energy_update_decreasing is False
                stopping_index = np.argmin(energy_update_decreasing)
                stopping_criterion = StoppingCriterion.ENERGY_UPDATE_STOPS_DECREASING

        state = OFState(
            mol=self.mol,
            coeffs=self.coeffs[stopping_index],
            energy=self.energy[stopping_index],
        )
        state.stopping_index = stopping_index
        self.stopping_index = stopping_index
        state.stopping_criterion = stopping_criterion
        return state
