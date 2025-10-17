"""MAE of energy, MAE for gradient, MAE for proj_minao."""
import torch
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool
from torchmetrics import Metric

from mldft.ml.data.components.of_data import OFData
from mldft.ml.models.components.sample_weighers import (
    ConstantSampleWeigher,
    HasEnergyLabelSampleWeigher,
    SampleWeigher,
)


class PerSampleAbsoluteErrorMetric(Metric):
    """Base class for metrics that calculate the absolute error per sample."""

    AVERAGING_MODES = ["per molecule", "per electron"]
    DEFAULT_SAMPLE_WEIGHER = None

    def __init__(self, mode="per molecule", sample_weigher: SampleWeigher | None = "default"):
        """
        Args:
            mode: The averaging mode, either "per molecule" or "per electron".
            sample_weigher (SampleWeigher, optional): Sample weigher to be used.
                Defaults to :attr:`DEFAULT_SAMPLE_WEIGHER`.
        """
        super().__init__()
        assert mode in self.AVERAGING_MODES, f"mode must be one of {self.AVERAGING_MODES}"
        self.mode = mode
        if sample_weigher == "default":
            assert (
                self.DEFAULT_SAMPLE_WEIGHER is not None
            ), f"DEFAULT_SAMPLE_WEIGHER is not set for class {self.__class__.__name__}"
            sample_weigher = self.DEFAULT_SAMPLE_WEIGHER
        self.sample_weigher = (
            sample_weigher if sample_weigher is not None else ConstantSampleWeigher()
        )

        self.add_state("weighted_molecule_count", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_absolute_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "sum_per_electron_absolute_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update_with_errors(self, batch: Batch, errors: torch.Tensor):
        """Updates the hidden states for a batch and corresponding errors.

        Args:
            batch: The OFData object, used for updating the molecule and electron count.
            errors: The errors that should be accumulated. The shape should either be (batch_size,) or (n_basis,), in
                the latter case the errors per basis function are summed up to get the error per molecule.
        """
        assert errors.ndim == 1, f"{errors.shape}"
        assert isinstance(batch, OFData)
        errors = errors.abs()
        if errors.shape == (batch.n_basis,):
            # we got an error per basis function, have to sum it up to get the error per molecule
            errors = global_add_pool(errors, batch.coeffs_batch)
        assert errors.shape[0] == batch.batch_size, f"{errors.size} != {batch.batch_size}"

        sample_weights = self.sample_weigher.get_weights(batch)

        self.sum_absolute_error += (errors * sample_weights).sum()
        self.weighted_molecule_count += sample_weights.sum()

        # this assumes neutral molecules, as this is technically the number of protons
        electrons_per_molecule = global_add_pool(batch.atomic_numbers, batch.atomic_numbers_batch)

        self.sum_per_electron_absolute_error += (
            errors * sample_weights / electrons_per_molecule
        ).sum()

    def compute_per_molecule(self):
        """Calculates the mean error per molecule."""
        return self.sum_absolute_error / self.weighted_molecule_count

    def compute_per_electron(self):
        """Calculates the mean error per electron."""
        return self.sum_per_electron_absolute_error / self.weighted_molecule_count

    def compute(self):
        """Calculates the mean error, based on the averaging mode."""
        if self.mode == "per molecule":
            return self.compute_per_molecule()
        elif self.mode == "per electron":
            return self.compute_per_electron()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")


class MAEGradient(PerSampleAbsoluteErrorMetric):
    r"""Mean Absolute Error - Metric of the gradient of the kinetic energy averaged over the number of molecules.

    For the default ``mode="per molecule"``, the error is averaged as

    .. math::

        \text{MAE_Gradint} = \frac{1}{\text{n_molecules}}\sum_d\sum_k ||
        \left( \textbf{I}-\frac{\textbf{w}^{(d)}{\textbf{w}^{(d)}}^T}{{\textbf{w}^{(d)}}^T
        \textbf{w}^{(d)}}\right)\left(\nabla_\textbf{p} T_{S,\Theta}(\textbf{p}^{(d,k)},\mathcal{M^{(d)}})-
        \nabla_\textbf{p} T_S^{(d,k)}\right)||.

    For ``mode="per electron"``, the error is averaged as

    .. math::

        \text{WMAE_Gradint} = \frac{1}{\text{n_molecules}}\sum_d \frac{1}{\text{n_electrons}^{(d)}}\sum_k ||
        \left( \textbf{I}-\frac{\textbf{w}^{(d)}{\textbf{w}^{(d)}}^T}{{\textbf{w}^{(d)}}^T
        \textbf{w}^{(d)}}\right)\left(\nabla_\textbf{p} T_{S,\Theta}(\textbf{p}^{(d,k)},\mathcal{M^{(d)}})
        -\nabla_\textbf{p} T_S^{(d,k)}\right)||.

    """

    DEFAULT_SAMPLE_WEIGHER = (
        HasEnergyLabelSampleWeigher()
    )  # has gradient label <=> has energy label

    def update(
        self,
        batch: Batch,
        pred_energy: torch.Tensor,
        projected_gradient_difference: torch.Tensor,
        pred_diff: torch.Tensor,
    ):
        """Computes the error and updates the hidden states."""
        error = projected_gradient_difference
        self.update_with_errors(batch, error)


class MAEEnergy(PerSampleAbsoluteErrorMetric):
    r"""Metric for the mean absolute error of the kinetic energy. When ``mode="per molecule"``, the
    error is averaged as.

    .. math::

        \text{MAE_Energy} = \frac{1}{\text{n_molecules}}\sum_d\sum_k | T_{S,\Theta}(\textbf{p}^{(d,k)},\mathcal{M^{(d)}})-T_S^{(d,k)}|.
    """

    DEFAULT_SAMPLE_WEIGHER = HasEnergyLabelSampleWeigher()

    def update(
        self,
        batch: Batch,
        pred_energy: torch.Tensor,
        projected_gradient_difference: torch.Tensor,
        pred_diff: torch.Tensor,
    ):
        error = pred_energy - batch.energy_label
        self.update_with_errors(batch, error)


class MAEInitialGuess(PerSampleAbsoluteErrorMetric):
    """MAE of the initial guess delta coefficients."""

    DEFAULT_SAMPLE_WEIGHER = ConstantSampleWeigher()

    def update(
        self,
        batch: Batch,
        pred_energy: torch.Tensor,
        projected_gradient_difference: torch.Tensor,
        pred_diff: torch.Tensor,
    ):
        """Computes the error and updates the hidden states."""
        error = pred_diff - (batch.coeffs - batch.ground_state_coeffs)
        self.update_with_errors(batch, error)
