import functools
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Type

import numpy as np
import torch
from scipy.optimize import LinearConstraint, minimize
from torch_geometric.nn import global_add_pool
from tqdm import tqdm

from mldft.ml.data.components.of_data import OFData
from mldft.ml.models.components.loss_function import project_gradient
from mldft.ofdft.energies import Energies


class Optimizer(ABC):
    """Base class for optimization algorithms for density optimization."""

    @abstractmethod
    def optimize(
        self,
        sample: OFData,
        energy_functional: Callable[[OFData], tuple[Energies, torch.Tensor]],
        callback: Callable | None = None,
        disable_pbar: bool = False,
    ) -> tuple[Energies, bool]:
        """Perform density optimization.

        Args:
            sample: The OFData containing the initial coefficients.
            energy_functional: Callable which returns the energy and gradient vector.
            callback: Optional callback function.
            disable_pbar: Whether to disable the progress bar.

        Returns:
            Final energy.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """Return a string representation of the optimizer."""
        name = self.__class__.__name__
        settings = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"{name}({settings})"


def get_pbar_str(sample: OFData, energy: Energies, gradient_norm: float) -> str:
    """Return a string for the tqdm progress bar.

    Args:
        sample: The OFData containing the current coefficients. If the ground state energy is
            available, the energy difference to the ground state is calculated.
        energy: The current energy.
        gradient_norm: The norm of the gradient vector.

    Returns:
        A string for the tqdm progress bar.
    """
    if "of_labels/energies/ground_state_e_electron" in sample:
        ground_state_electronic_energy = sample["of_labels/energies/ground_state_e_electron"]
        if isinstance(ground_state_electronic_energy, torch.Tensor):
            ground_state_electronic_energy = ground_state_electronic_energy.item()
        delta_e = energy.electronic_energy - ground_state_electronic_energy
        delta_e *= 1e3  # convert to mHa
        return f"ΔE={delta_e:.3e} mHa, grad_norm={gradient_norm:.3e}"
    else:
        return f"E_elec={energy.electronic_energy:.6f} Ha, grad_norm={gradient_norm:.3e}"


class GradientDescent(Optimizer):
    """Simple gradient descent optimizer."""

    def __init__(self, learning_rate: float, convergence_tolerance: float, max_cycle: int):
        """Initialize the gradient descent optimizer.

        Args:
            max_cycle: Maximum number of optimization cycles.
            convergence_tolerance: Optimization stops if the gradient norm is below this value.
            learning_rate: The learning rate.
        """
        self.learning_rate = learning_rate
        self.convergence_tolerance = convergence_tolerance
        self.max_cycle = max_cycle

    def optimize(
        self,
        sample: OFData,
        energy_functional: Callable,
        callback: Callable | None = None,
        disable_pbar: bool = False,
    ) -> tuple[Energies, bool]:
        """Perform gradient descent optimization."""
        converged = False
        for cycle in (
            pbar := tqdm(
                range(self.max_cycle),
                leave=False,
                dynamic_ncols=True,
                position=int(os.getenv("DENOP_PID", 0)),
                disable=disable_pbar,
            )
        ):
            energy, gradient_vector = energy_functional(sample)
            projected_gradient = project_gradient(gradient_vector, sample)
            gradient_norm = torch.norm(projected_gradient).item()

            pbar.set_description(get_pbar_str(sample, energy, gradient_norm))
            if callable(callback):
                coeffs = sample.coeffs  # for callback
                learning_rate = self.learning_rate
                callback(locals())

            if gradient_norm < self.convergence_tolerance:
                converged = True
                break

            sample.coeffs -= self.learning_rate * projected_gradient
        pbar.close()
        return energy, converged


class TorchOptimizer(Optimizer):
    """Wrapper for torch optimizers to be used in the optimization loop."""

    def __init__(
        self,
        torch_optimizer: Type[torch.optim.Optimizer],
        convergence_tolerance: float,
        max_cycle: int,
        **optimizer_kwargs,
    ):
        """Initialize the torch optimizer.

        Args:
            torch_optimizer: The torch optimizer to use. To be able to apply the optimizer with
                hydra, the class is partially applied without any arguments.
            convergence_tolerance: Optimization stops if the gradient norm is below this value.
            max_cycle: Maximum number of optimization cycles.
            optimizer_kwargs: Additional keyword arguments for the optimizer.
        """
        self.torch_optimizer = torch_optimizer
        self.convergence_tolerance = convergence_tolerance
        self.max_cycle = max_cycle
        self.optimizer_kwargs = optimizer_kwargs

    def optimize(
        self,
        sample: OFData,
        energy_functional: Callable,
        callback: Callable | None = None,
        disable_pbar: bool = False,
    ) -> tuple[Energies, bool]:
        """Optimization loop for a torch optimizer."""
        parameters = sample.coeffs.clone()
        optimizer = self.torch_optimizer([parameters], **self.optimizer_kwargs)

        converged = False
        for cycle in (
            pbar := tqdm(
                range(self.max_cycle),
                leave=False,
                dynamic_ncols=True,
                position=int(os.getenv("DENOP_PID", 0)),
                disable=disable_pbar,
            )
        ):
            energy, gradient_vector = energy_functional(sample)
            projected_gradient = project_gradient(gradient_vector, sample)
            gradient_norm = torch.norm(projected_gradient).item()

            pbar.set_description(get_pbar_str(sample, energy, gradient_norm))
            if callable(callback):
                coeffs = sample.coeffs  # for callback
                if "lr" in self.optimizer_kwargs:
                    learning_rate = self.optimizer_kwargs["lr"]
                else:
                    learning_rate = 0
                callback(locals())

            if gradient_norm < self.convergence_tolerance:
                converged = True
                break

            optimizer.zero_grad()
            parameters.grad = projected_gradient  # FIXME: could also use normal gradient
            optimizer.step()
            update = parameters - sample.coeffs
            # projecting the update is important for, e.g., the Adam optimizer
            sample.coeffs += project_gradient(update, sample)
            # update parameters but clone to not update sample.coeffs in optimizer step
            parameters.data = sample.coeffs.detach().clone()

        pbar.close()
        return energy, converged

    def __str__(self) -> str:
        """Return a string representation of the optimizer.

        This method is overwritten since the torch optimizer can either be a class via direct
        instantiation or via hydra with a partial.
        """
        if isinstance(self.torch_optimizer, functools.partial):  # if called with config
            name = self.torch_optimizer.func.__name__
        else:
            name = self.torch_optimizer.__class__.__name__
        settings = ", ".join(f"{k}={v}" for k, v in self.optimizer_kwargs.items())
        return f"{name}({settings})"


class VectorAdam(Optimizer):
    """Equivariant version of the Adam optimizer."""

    def __init__(
        self,
        max_cycle: int,
        learning_rate: float,
        convergence_tolerance: float,
        betas: tuple[float, float] = (0.9, 0.999),
        epsilon: float = 1e-8,
    ):
        """Initialize the equivariant version of the Adam optimizer.

        Args:
            max_cycle: Maximum number of optimization cycles.
            learning_rate: Learning rate function.
            convergence_tolerance: Optimization stops if the gradient norm is below this value.
            betas: Exponential decay rates for the moment estimates.
            epsilon: Small value to avoid division by zero.
        """
        self.max_cycle = max_cycle
        self.learning_rate = learning_rate
        self.convergence_tolerance = convergence_tolerance
        self.betas = betas
        self.epsilon = epsilon

    def optimize(
        self,
        sample: OFData,
        energy_functional: Callable,
        callback: Callable | None = None,
        disable_pbar: bool = False,
    ) -> tuple[Energies, bool]:
        """Perform equivariant VectorAdam optimization."""
        converged = False
        m = torch.zeros_like(sample.coeffs)
        v = torch.zeros_like(sample.coeffs)
        beta1, beta2 = self.betas

        shell_beginning_mask = sample.basis_info.shell_beginning_mask
        shell_beginning_mask = torch.as_tensor(shell_beginning_mask, device=sample.coeffs.device)
        shell_indices = torch.cumsum(shell_beginning_mask[sample.basis_function_ind], dim=0) - 1

        for cycle in (
            pbar := tqdm(
                range(self.max_cycle),
                dynamic_ncols=True,
                position=int(os.getenv("DENOP_PID", 0)),
                disable=disable_pbar,
            )
        ):
            energy, gradient_vector = energy_functional(sample)
            projected_gradient = project_gradient(gradient_vector, sample)
            gradient_norm = torch.norm(projected_gradient).item()

            m = beta1 * m + (1 - beta1) * projected_gradient
            # squared gradient per shell
            v_per_irrep = global_add_pool(projected_gradient**2, shell_indices)
            v = beta2 * v + (1 - beta2) * v_per_irrep[shell_indices]
            m_hat = m / (1 - beta1 ** (cycle + 1))
            v_hat = v / (1 - beta2 ** (cycle + 1))
            update = self.learning_rate * m_hat / (torch.sqrt(v_hat) + self.epsilon)
            update = project_gradient(update, sample)

            pbar.set_description(get_pbar_str(sample, energy, gradient_norm))
            if callable(callback):
                coeffs = sample.coeffs  # for callback
                learning_rate = self.learning_rate
                callback(locals())

            if gradient_norm < self.convergence_tolerance:
                converged = True
                break

            sample.coeffs -= update
        pbar.close()
        return energy, converged


def scipy_functional(
    coeffs: np.ndarray,
    sample: OFData,
    energy_functional: Callable,
    convergence_tolerance: float,
    use_projected_gradient: bool,
    pbar: tqdm,
    callback: Callable | None = None,
) -> tuple[float, np.ndarray]:
    """Functional for scipy optimizers.

    Make the energy functional compatible with scipy optimizers.

    Args:
        coeffs: Input coefficients.
        sample: OFData containing the basis functions and integrals.
        energy_functional: Callable which returns the energy and gradient vector.
        convergence_tolerance: Optimization stops if the gradient norm is below this value.
        use_projected_gradient: Whether to use the projected gradient for the optimization step.
        pbar: tqdm progress bar.
        callback: Optional callback function.

    Returns:
        Energy and gradient vector.
    """
    sample.coeffs = torch.tensor(coeffs, dtype=torch.float64, device=sample.coeffs.device)
    energy, gradient = energy_functional(sample)
    projected_gradient = project_gradient(gradient, sample)
    gradient_norm = torch.norm(projected_gradient).item()

    if callable(callback):
        coeffs = torch.tensor(coeffs)
        learning_rate = 0  # needed for callback
        callback(locals())
    pbar.set_description(get_pbar_str(sample, energy, gradient_norm))

    if use_projected_gradient:
        gradient_np = projected_gradient.cpu().numpy()
    else:
        gradient_np = gradient.cpu().numpy()

    if gradient_norm < convergence_tolerance:
        raise StopIteration
    pbar.close()
    return energy.electronic_energy, gradient_np


class SLSQP(Optimizer):
    """Wrapper for the SLSQP (Sequential Least Squares Programming) optimizer from scipy."""

    def __init__(
        self,
        max_cycle: int,
        convergence_tolerance: float,
        grad_scale: float,
        use_projected_gradient: bool = True,
    ):
        """Initialize the SLSQP optimizer.

        Args:
            max_cycle: Maximum number of optimization cycles. Note that this is not the same as the
                number of functional evaluations.
            convergence_tolerance: Optimization stops if the gradient norm is below this value.
            grad_scale: Scaling factor for the gradient vector.
            use_projected_gradient: Whether to use the projected gradient for the optimization
                step.
        """
        self.max_cycle = max_cycle
        self.convergence_tolerance = convergence_tolerance
        self.grad_scale = grad_scale
        self.use_projected_gradient = use_projected_gradient

    def optimize(
        self,
        sample: OFData,
        energy_functional: Callable,
        callback: Callable | None = None,
        disable_pbar: bool = False,
    ) -> tuple[Energies, bool]:
        """Perform density optimization using the SLSQP optimizer."""
        converged = False
        normalization_vector = sample.dual_basis_integrals.cpu().numpy()
        n_electron = sample.n_electron
        pbar = tqdm(
            range(self.max_cycle),
            leave=False,
            dynamic_ncols=True,
            position=int(os.getenv("DENOP_PID", 0)),
            disable=disable_pbar,
        )
        iterator = iter(pbar)

        functional = functools.partial(
            scipy_functional,
            sample=sample,
            energy_functional=energy_functional,
            convergence_tolerance=self.convergence_tolerance,
            use_projected_gradient=self.use_projected_gradient,
            pbar=pbar,
            callback=callback,
        )

        # decrease the gradient to make the optimization more stable
        def functional_scaled(coeffs: np.ndarray) -> tuple[float, np.ndarray]:
            energy, gradient = functional(coeffs)
            return energy, self.grad_scale * gradient

        def electron_number_constraint(coeffs: np.ndarray) -> float:
            return np.dot(coeffs, normalization_vector) - n_electron

        def derivative_electron_number_constraint(_) -> np.ndarray:
            return normalization_vector

        constraint = {
            "type": "eq",
            "fun": electron_number_constraint,
            "jac": derivative_electron_number_constraint,
        }

        try:
            minimize(
                functional_scaled,
                x0=sample.coeffs.cpu().numpy(),
                method="SLSQP",
                jac=True,
                constraints=constraint,
                options={"maxiter": self.max_cycle, "ftol": 0},  # disable tolerance
                callback=lambda _: next(iterator),
            )
        except StopIteration:
            converged = True
            pass  # StopIteration is raised when the gradient norm is below the tolerance
        pbar.close()
        return energy_functional(sample)[0], converged


class TrustRegionConstrained(Optimizer):
    """Wrapper for the trust region constrained optimizer from scipy."""

    def __init__(
        self,
        max_cycle: int,
        convergence_tolerance: float,
        initial_tr_radius: float,
        initial_constr_penalty: float,
        use_projected_gradient: bool = True,
    ):
        """Initialize the trust region constrained optimizer.

        Args:
            convergence_tolerance:
            max_cycle: Maximum number of optimization cycles.
            convergence_tolerance: Optimization stops if the gradient norm is below this value.
            initial_tr_radius: Initial trust radius. Affects the size of the first steps.
            initial_constr_penalty: Initial constraint penalty.
            use_projected_gradient: Whether to use the projected gradient for the optimization step.
        """
        self.max_cycle = max_cycle
        self.convergence_tolerance = convergence_tolerance
        self.initial_tr_radius = initial_tr_radius
        self.initial_constr_penalty = initial_constr_penalty
        self.use_projected_gradient = use_projected_gradient

    def optimize(
        self,
        sample: OFData,
        energy_functional: Callable,
        callback: Callable | None = None,
        disable_pbar: bool = False,
    ) -> tuple[Energies, bool]:
        """Perform trust region constrained optimization."""
        converged = False
        normalization_vector = sample.dual_basis_integrals.cpu().numpy()
        n_electron = sample.n_electron
        pbar = tqdm(
            range(self.max_cycle),
            leave=False,
            dynamic_ncols=True,
            position=int(os.getenv("DENOP_PID", 0)),
            disable=disable_pbar,
        )
        iterator = iter(pbar)

        functional = functools.partial(
            scipy_functional,
            sample=sample,
            energy_functional=energy_functional,
            convergence_tolerance=self.convergence_tolerance,
            use_projected_gradient=self.use_projected_gradient,
            pbar=pbar,
            callback=callback,
        )
        constraint = LinearConstraint(normalization_vector, n_electron, n_electron)

        def scipy_callback(*_):
            next(iterator)

        try:
            minimize(
                functional,
                x0=sample.coeffs.cpu().numpy(),
                method="trust-constr",
                jac=True,
                tol=0,  # disable tolerance
                constraints=constraint,
                options={
                    "maxiter": self.max_cycle,
                    "initial_tr_radius": self.initial_tr_radius,
                    "initial_constr_penalty": self.initial_constr_penalty,
                },
                callback=scipy_callback,
            )
        except StopIteration:
            converged = True
            pass  # StopIteration is raised when the gradient norm is below the tolerance
        pbar.close()
        return energy_functional(sample)[0], converged


DEFAULT_DENSITY_OPTIMIZER = TorchOptimizer(
    torch.optim.SGD,
    convergence_tolerance=1e-4,
    max_cycle=1000,
    lr=1e-3,
    momentum=0.9,
)
