import pytest

from mldft.ofdft.callbacks import ConvergenceCallback, ConvergenceCriterion
from mldft.ofdft.energies import Energies
from mldft.ofdft.ofstate import StoppingCriterion

test_energies = [
    [9, 0, 1, 5],
    [20, 19, 2, 1, 0],
    [1, 0],
    [1, 0, 1, 2],
    [10, 5, 4, 1],
    [90, 50, 2, 1],
    [1, 0, 1, 2],
]
test_gradients = [
    [0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0],
    [2, 1, 0, 1],
    [3, 2, 1, 0],
    [3, 2, 1, 0],
    [0, 1, 2, 3],
]
test_convergence_criteria = [
    ConvergenceCriterion.MIN_E_STEP,
    ConvergenceCriterion.MIN_E_STEP,
    ConvergenceCriterion.MOFDFT,
    ConvergenceCriterion.MOFDFT,
    ConvergenceCriterion.MOFDFT,
    ConvergenceCriterion.MOFDFT,
    ConvergenceCriterion.MOFDFT,
]

expected_indices = [1, 0, 0, 2, 1, 0, 0]
expected_criteria = [
    StoppingCriterion.ENERGY_UPDATE_GLOBAL_MINIMUM,
    StoppingCriterion.ENERGY_UPDATE_GLOBAL_MINIMUM,
    StoppingCriterion.GRADIENT_STOPS_DECREASING,
    StoppingCriterion.GRADIENT_STOPS_DECREASING,
    StoppingCriterion.ENERGY_UPDATE_STOPS_DECREASING,
    StoppingCriterion.ENERGY_UPDATE_STOPS_DECREASING,
    StoppingCriterion.GRADIENT_STOPS_DECREASING,
]

test_cases = [
    (energies, gradients, small_scale, expected_index, expected_criterion)
    for energies, gradients, small_scale, expected_index, expected_criterion in zip(
        test_energies,
        test_gradients,
        test_convergence_criteria,
        expected_indices,
        expected_criteria,
    )
]


@pytest.mark.parametrize(
    "energies, gradients, small_scale, expected_index,expected_criterion",
    test_cases,
)
def test_convergence_criterion(
    energies, gradients, small_scale, expected_index, expected_criterion
):
    """Compare the convergence result to results obtained by hand."""
    callback = ConvergenceCallback()

    list_of_energies = []
    for energy in energies:
        # build energy object that only contains kinetic energy
        energies_ = Energies(kinetic=energy, hartree=0, xc=0, nuclear_attraction=0)
        energies_["nuclear_repulsion"] = 0
        list_of_energies.append(energies_)

    callback.energy = list_of_energies
    callback.coeffs = [0] * len(energies)
    callback.gradient_norm = gradients

    state = callback.get_convergence_result(small_scale)

    assert state.stopping_index == expected_index
    assert state.energy.total_energy == energies[expected_index]
    assert state.stopping_criterion == expected_criterion


def test_convergence_criterion_raises():
    """Test that the convergence criterion raises ValueErrors."""
    callback = ConvergenceCallback()

    with pytest.raises(ValueError):
        # energies values
        callback.get_convergence_result(False)

    callback = ConvergenceCallback(energy=[Energies(), Energies()])

    with pytest.raises(ValueError):
        # length of energies, coeffs and gradient_norm does not match
        callback.get_convergence_result(False)
