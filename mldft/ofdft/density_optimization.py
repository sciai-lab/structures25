from typing import Callable, Optional

import numpy as np
import torch
from loguru import logger
from pyscf import gto

from mldft.ml.data.components.basis_transforms import transform_tensor_with_sample
from mldft.ml.data.components.of_data import OFData, Representation
from mldft.ml.models.mldft_module import MLDFTLitModule
from mldft.ofdft.callbacks import ConvergenceCallback, ConvergenceCriterion
from mldft.ofdft.energies import Energies
from mldft.ofdft.functional_factory import FunctionalFactory
from mldft.ofdft.initial_guess import (
    dumb_guess,
    hueckel_guess,
    label_guess,
    minao_guess,
    perturbed_label_guess,
    proj_minao_guess,
)
from mldft.ofdft.optimizer import Optimizer
from mldft.utils.grids import grid_setup
from mldft.utils.pyscf_pretty_print import mole_to_sum_formula
from mldft.utils.sad_guesser import SADGuesser


def initial_guess(
    sample: OFData,
    mol: gto.Mole,
    initial_guess_str: str,
    normalize_initial_guess: bool = True,
    ks_basis: str = "6-31G(2df,p)",
    proj_minao_module: MLDFTLitModule | None = None,
    sad_guess_kwargs: dict | None = None,
) -> torch.Tensor:
    """Return the initial guess for the coefficients.

    Args:
        sample: The OFData object.
        mol: The molecule.

    Returns:
        torch.Tensor: The initial guess for the coefficients.

    Raises:
        KeyError: If the initial guess is not in INITIAL_GUESS (e.g. implemented guess).
    """
    if initial_guess_str == "dumb":
        return dumb_guess(sample, mol)
    elif initial_guess_str == "label":
        return label_guess(sample)
    elif initial_guess_str.startswith("perturbed_label"):
        scale = float(initial_guess_str.split("_")[-1])
        return perturbed_label_guess(sample, scale)
    elif initial_guess_str == "hueckel":
        return hueckel_guess(sample, mol, ks_basis, normalize=normalize_initial_guess)
    elif initial_guess_str == "minao":
        return minao_guess(sample, mol, ks_basis, normalize=normalize_initial_guess)
    elif initial_guess_str == "proj_minao":
        assert proj_minao_module is not None, "A lightning Module must be provided for proj_minao."
        return proj_minao_guess(
            sample,
            mol,
            proj_minao_module,
            ks_basis,
            normalize=normalize_initial_guess,
        )
    elif initial_guess_str == "sad":
        guesser = SADGuesser.from_dataset_statistics(**sad_guess_kwargs)
        guesser = guesser.to(device=sample.coeffs.device)

        # this is a bit of a hack, but we need the untransformed basis integrals, and this is one way to get them
        untransformed_sample = sample.clone()
        untransformed_sample.dual_basis_integrals = transform_tensor_with_sample(
            untransformed_sample,
            untransformed_sample.dual_basis_integrals,
            Representation.DUAL_VECTOR,
            invert=True,
        )
        coeffs_guess = guesser(untransformed_sample)
        coeffs_guess = transform_tensor_with_sample(sample, coeffs_guess, Representation.VECTOR)
        return coeffs_guess
    elif initial_guess_str == "sad_transformed":
        # compute the SAD guess in the (potentially transformed) basis of the sample
        # this seems to work well, except for global natural reparametrization
        guesser = SADGuesser.from_dataset_statistics(**sad_guess_kwargs)
        guesser = guesser.to(device=sample.coeffs.device)
        return guesser(sample)
    else:
        raise KeyError(f"Initial guess {initial_guess_str} not recognized.")


def compute_density_optimization_metrics(
    callback: ConvergenceCallback,
    ground_state_energy: Energies,
    convergence_criterion: str | ConvergenceCriterion,
) -> dict:
    """Compute metrics for the density optimization process."""
    metric_dict = {}
    total_energies = np.asarray([e.total_energy for e in callback.energy])
    converged_state = callback.get_convergence_result(convergence_criterion)
    stopping_index = converged_state.stopping_index
    signed_energy_error = total_energies[stopping_index] - ground_state_energy.total_energy
    metric_dict["gs_signed_energy_error"] = signed_energy_error
    metric_dict["gs_abs_energy_error"] = np.abs(signed_energy_error)
    energy_differences = total_energies - ground_state_energy.total_energy
    metric_dict["min_energy_error"] = energy_differences.min()
    metric_dict["gs_l2_norm"] = callback.l2_norm[stopping_index]
    metric_dict["minimum_l2_norm"] = min(callback.l2_norm)
    metric_dict["stopping_index"] = stopping_index
    metric_dict["gs_gradient_norm"] = callback.gradient_norm[stopping_index]
    metric_dict["min_gradient_norm"] = min(callback.gradient_norm)
    return metric_dict


def density_optimization(
    sample: OFData,
    mol: gto.Mole,
    optimizer: Optimizer,
    func_factory: FunctionalFactory,
    callback: ConvergenceCallback | None = None,
    initialization: str | torch.Tensor = "minao",
    max_xc_memory: int = 4000,
    normalize_initial_guess: bool = True,
    ks_basis: str = "6-31G(2df,p)",
    proj_minao_module: Optional["MLDFTLitModule"] = None,
    sad_guess_kwargs: dict | None = None,
    disable_printing: bool = False,
    disable_pbar: bool = None,
) -> tuple[Energies, torch.Tensor, bool, Callable]:
    """Perform density optimization for a given sample and molecule.

    Args:
        sample: OFData sample object containing required tensors for the functional.
        mol: Molecule object used for the initial guess and building the grid.
        optimizer: Optimizer used for the density optimization process.
        func_factory: FunctionalFactory object used to construct the energy functional.
        callback: ConvergenceCallback object to store the optimization history.
        initialization:  Which initial guess to use.
        max_xc_memory: Maximum memory for the XC functional.
        normalize_initial_guess: Whether to normalize the initial guess to the correct number of electrons.
        ks_basis: Basis set to use for the initial guess.
        proj_minao_module: Lightning Module used for the proj_minao initial guess. Only required if initialization is 'proj_minao'.
        sad_guess_kwargs: Dictionary of arguments for the SAD guesser. Only required if initialization is 'sad' or 'sad_transformed'.
        disable_printing: Whether to disable printing (useful when running density optimization during training with the
            rich progress bar, as output becomes messy).
        disable_pbar: Whether to disable the progress bar. If None, it will use the value of disable_printing.

    Returns:
        Energies: Energies object containing the total energy and its components.
        torch.Tensor: Final coefficients in the AO basis.
        bool: Whether the optimization converged.
    """
    if disable_pbar is None:
        disable_pbar = disable_printing
    #  Prepare aos first, otherwise sample.clones will be too large
    if hasattr(sample, "grid_level"):
        grid = grid_setup(mol, sample.grid_level, sample.grid_prune)
    else:
        grid = None
    if hasattr(sample, "ao"):
        ao = sample.ao
        # On older GPUs having the tensor in this format gives a 3x speed up for the xc functional
        if ao.device.type == "cuda" and torch.cuda.get_device_capability()[0] < 8:
            ao = ao.transpose(1, 2).contiguous().transpose(1, 2)
            # this leaves large unused memory blocks, so we free them
            torch.cuda.empty_cache()
        # delete from sample to allow for sample cloning
        sample.delete_item("ao")
    else:
        ao = None

    # Prepare initial guess and functional
    if isinstance(initialization, torch.Tensor):
        initial_coeffs = initialization
    else:
        initial_coeffs = initial_guess(
            sample,
            mol,
            initialization,
            normalize_initial_guess,
            ks_basis,
            proj_minao_module,
            sad_guess_kwargs,
        )
    sample.coeffs = initial_coeffs
    nelectron_guess = sample.dual_basis_integrals @ initial_coeffs
    if not np.isclose(nelectron_guess.item(), mol.nelectron, atol=1e-2):
        logger.warning(
            f"Guess normalized to {nelectron_guess} electrons, but {mol.nelectron} expected."
        )
    energy_functional = func_factory.construct(
        mol,
        sample.coulomb_matrix,
        sample.nuclear_attraction_vector,
        grid,
        ao,
        max_xc_memory,
    )
    # Optimization
    final_energies, converged = optimizer.optimize(
        sample, energy_functional, callback, disable_pbar=disable_pbar
    )
    final_coeffs = transform_tensor_with_sample(
        sample, sample.coeffs, Representation.VECTOR, invert=True
    )

    if not disable_printing:
        logger.info(
            f"{'Summary':<11}{mole_to_sum_formula(mol, use_subscript=True):<16}"
            f"E={final_energies.total_energy} Ha, "
            f"converged: {converged}"
        )

    return final_energies, final_coeffs, converged, energy_functional


def density_optimization_with_label(
    sample: OFData,
    mol: gto.Mole,
    optimizer: Optimizer,
    func_factory: FunctionalFactory,
    callback: ConvergenceCallback | None,
    max_xc_memory: int = 4000,
    initial_guess_str: str = "minao",
    normalize_initial_guess: bool = True,
    ks_basis: str = "6-31G(2df,p)",
    proj_minao_module: Optional["MLDFTLitModule"] = None,
    sad_guess_kwargs: dict | None = None,
    convergence_criterion: str | ConvergenceCriterion = "last_iter",
    disable_printing: bool = False,
) -> tuple[dict, ConvergenceCallback, Energies, Callable]:
    """Perform density optimization for a given sample and molecule.

    Args:
        sample: OFData sample object containing required tensors for the functional.
        mol: Molecule object used for the initial guess and building the grid.
        optimizer: Optimizer used for the density optimization process.
        initial_guess_str: Which initial guess to use.
        normalize_initial_guess: Whether to normalize the initial guess to the correct number of electrons.
        ks_basis: Basis set to use for the initial guess.
        proj_minao_module: Lightning Module used for the proj_minao initial guess. Only required if initial_guess_str is 'proj_minao'.
        sad_guess_kwargs: Dictionary of arguments for the SAD guesser. Only required if initial_guess_str is 'sad' or 'sad_transformed'.
        convergence_criterion: Criterion to determine the ground state of the optimization process.

    Returns:
        metric_dict: Dictionary containing metrics of the density optimization.
        callback: ConvergenceCallback object containing the optimization history for plotting and further analysis.
        energies_label: The ground state label of the molecule.
        energy_functional: The constructed energy functional.
    """
    _, _, _, energy_functional = density_optimization(
        sample,
        mol,
        optimizer,
        func_factory,
        callback=callback,
        initialization=initial_guess_str,
        normalize_initial_guess=normalize_initial_guess,
        max_xc_memory=max_xc_memory,
        ks_basis=ks_basis,
        proj_minao_module=proj_minao_module,
        sad_guess_kwargs=sad_guess_kwargs,
        disable_printing=True,
        disable_pbar=disable_printing,
    )
    energies_label = func_factory.get_energies_label(sample, basis_info=sample.basis_info)
    # Compute metrics
    metric_dict = compute_density_optimization_metrics(
        callback, energies_label, convergence_criterion
    )
    if not disable_printing:
        logger.info(
            f"{'Summary':<11}{mole_to_sum_formula(mol, use_subscript=True):<16}"
            f"ΔE={metric_dict['gs_signed_energy_error'] * 1000:6.2g} mHa, "
            f"ρNorm={metric_dict['gs_l2_norm']:6.2g}, "
            f"gNorm={metric_dict['gs_gradient_norm']:7.2e}"
        )
    return metric_dict, callback, energies_label, energy_functional
