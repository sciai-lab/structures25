"""Module containing the loss functions for the ML-DFT model."""
from typing import Mapping

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn.pool import global_add_pool

from mldft.ml.data.components.of_data import OFData
from mldft.ml.models.components.sample_weighers import SampleWeigher


def project_gradient(gradient: torch.Tensor, batch_or_sample: OFData) -> torch.Tensor:
    r"""Calculates the projected gradient.

    Note that when using non-orthogonal transformations we need the dual vector of :math:`w`, written as :math:`w^*`,
    since it will transform different to :math:`w^T`. For orthogonal transformations we have :math:`w^* = w^T`.
    We get rid of the matrix multiplications by rewriting the equation as follows:

    .. math::

        \left( \boldsymbol{I} - \frac{\boldsymbol{w}\boldsymbol{w}^T}{\boldsymbol{w}^T \boldsymbol{w}} \right)
        \nabla_\boldsymbol{p} T =
        \nabla_\boldsymbol{p} T - \boldsymbol{w}
        \frac{\boldsymbol{w}^T \nabla_\boldsymbol{p} T}
        {{\boldsymbol{w}^T \boldsymbol{w}}}

    Args:
        gradient (torch.Tensor): The predicted gradients
        batch_or_sample (OFData): The OFData object (can be a batch or a single sample) containing the basis integrals.

    Returns:
        torch.Tensor: The projected gradient
    """
    if isinstance(batch_or_sample, Batch):
        coeffs_batch = batch_or_sample.coeffs_batch
    else:
        coeffs_batch = None

    w_p = global_add_pool(batch_or_sample.dual_basis_integrals * gradient, coeffs_batch)
    w_w = global_add_pool(
        batch_or_sample.dual_basis_integrals * batch_or_sample.dual_basis_integrals, coeffs_batch
    )
    factor = w_p / w_w

    if isinstance(batch_or_sample, Batch):
        projected_gradient = (
            gradient
            - batch_or_sample.dual_basis_integrals
            * torch.repeat_interleave(factor, torch.bincount(batch_or_sample.coeffs_batch))
        )
    else:
        projected_gradient = gradient - batch_or_sample.dual_basis_integrals * factor

    return projected_gradient


def project_gradient_difference(pred_gradients: torch.Tensor, batch: OFData) -> torch.Tensor:
    r"""Calculates the projected gradient difference and the absolute projected gradient error.

    We get rid of the matrix multiplications by rewriting the equation as follows:

    .. math::

        \left( \boldsymbol{I} - \frac{\boldsymbol{w}\boldsymbol{w}^T}{\boldsymbol{w}^T \boldsymbol{w}} \right)
        \left(\nabla_\boldsymbol{p} T_{\mathrm{pred}} - \nabla_\boldsymbol{p} T \right) =
        \left(\nabla_\boldsymbol{p} T_{\mathrm{pred}}-\nabla_\boldsymbol{p} T\right) - \boldsymbol{w}
        \frac{\boldsymbol{w}^T \left(\nabla_\boldsymbol{p} T_{\text{pred}}-\nabla_\boldsymbol{p} T\right)}
        {{\boldsymbol{w}^T \boldsymbol{w}}}

    Args:
        pred_gradients (torch.Tensor): The predicted gradients
        batch (Batch): The batch object containing the target gradients and the basis integrals

    Returns:
        Unreduced tensor of projected differences
    """
    diff_gradients_batch = pred_gradients - batch.gradient_label
    return project_gradient(diff_gradients_batch, batch)


class SingleLossFunction(nn.Module):
    """Base class for loss functions that compute a single loss value."""

    def __init__(
        self,
        loss_function: nn.Module = None,
        sample_weigher: SampleWeigher = None,
        reduction: str = "mean",
    ):
        """Initialize the LossFunction by setting the loss function, weighing and reduction.

        Args:
            loss_function (nn.Module, optional): Loss function to be used. Defaults to nn.L1Loss().
                **Important** The loss function should not apply any reduction, as this is handled by the LossFunction
                after weighting the loss.
            sample_weigher (SampleWeigher, optional): Sample weigher to be used. Defaults to None.
            reduction (str, optional): Reduction type to be used. Defaults to "mean".
        """
        super().__init__()
        if loss_function is None:
            loss_function = nn.L1Loss(reduction="none")
        self.loss_function = loss_function
        self.sample_weigher = sample_weigher
        self.reduction = reduction

    def forward(self, batch: OFData, **kwargs) -> Tensor:
        """Get the per-sample losses and apply the sample weights, as defined by the
        sample_weigher.

        Args:
            batch (OFData): The batch object, used in the loss calculation and for the sample weights
            **kwargs: Additional arguments to be passed to the loss function

        Returns:
            Tensor: The scalar loss
        """
        loss = self.get_loss(batch, **kwargs)
        if self.sample_weigher is None:
            weighted_loss = loss
        else:
            weighted_loss = self.weigh_loss(batch, loss)
        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        else:
            raise ValueError(f"Unknown reduction type: {self.reduction}")

    def weigh_loss(self, batch: OFData, loss: Tensor) -> Tensor:
        """Function that applies weights to the loss.

        Has to be implemented by subclasses.
        """
        raise NotImplementedError

    def get_loss(self, batch: OFData, **kwargs) -> Tensor:
        """Function that computes the loss.

        Has to be implemented by subclasses.
        """
        raise NotImplementedError


class PerSampleWeightedPerSampleLossFunction(SingleLossFunction):
    """Base class for loss functions that compute a loss value per sample."""

    def weigh_loss(self, batch: OFData, loss: Tensor) -> Tensor:
        """Applies weights to the loss for each sample."""
        weights = self.sample_weigher.get_weights(batch)
        assert loss.shape == weights.shape, f"{loss.shape} != {weights.shape}"
        return loss * weights


class PerSampleWeightedPerCoeffLossFunction(SingleLossFunction):
    """Base class for loss functions that compute a loss value per basis function."""

    def weigh_loss(self, batch: OFData, loss: Tensor) -> Tensor:
        """Applies weights to the loss for each basis function."""
        weights = self.sample_weigher.get_weights(batch)
        weights = weights[batch.coeffs_batch]
        assert loss.shape == weights.shape, f"{loss.shape} != {weights.shape}"
        return (loss * weights).mean()


class EnergyLoss(PerSampleWeightedPerSampleLossFunction):
    """Calculates the loss between the predicted energy and the target energy."""

    def get_loss(self, batch: OFData, pred_energy: Tensor, **_) -> Tensor:
        """Computes the loss between the predicted energy and target energy.

        Args:
            batch (Batch): Batch object containing the target energy
            pred_energy (Tensor): Tensor containing the predicted energy

        Returns:
            Tensor: the loss for each sample in the batch
        """
        return self.loss_function(pred_energy, batch.energy_label)


class EnergyGradientLoss(PerSampleWeightedPerCoeffLossFunction):
    """Calculates the projected loss between the predicted gradients and the target gradients."""

    def get_loss(self, batch: Batch, projected_gradient_difference: Tensor, **_) -> Tensor:
        """Computes the loss to the projected difference of the predicted and ground truth
        gradient. Since the input is already the difference to the ground truth, the label is set
        to zero in the loss function call.

        Args:
            batch (Batch): Batch object containing the target energy
            projected_gradient_difference (Tensor): the difference of the predicted and real gradient, after projection

        Returns:
            Tensor: loss function of the input
        """
        return self.loss_function(
            projected_gradient_difference, torch.zeros_like(projected_gradient_difference)
        )


class CoefficientLoss(PerSampleWeightedPerCoeffLossFunction):
    """Calculates the loss between the predicted difference of coefficients and the target
    difference of coefficients."""

    def get_loss(self, batch: Batch, pred_diff: Tensor, **_) -> Tensor:
        """Returns the loss between the predicted difference of coefficients and the target
        difference of coefficients from the batch object.

        Args:
            batch (Batch): Batch object containing the target difference of coefficients
            pred_diff (Tensor): the predicted difference of the ground state coefficients and the current coefficients

        Returns:
            Tensor: loss between the predicted difference of coefficients and the target difference of coefficients,
                per basis function
        """
        return self.loss_function(pred_diff, batch.coeffs - batch.ground_state_coeffs)


class WeightedLoss(nn.Module):
    """Module used to combine multiple losses with different weights.

    The forward pass does not return a single scalar loss, but rather two dictionaries containing
    the weights and the losses for each individual component.
    """

    def __init__(self, **kwargs: Mapping[str, float | nn.Module]):
        """Initialize the WeightedLoss object by passing a mapping of str to loss function and
        weight.

        Args:
            **kwargs: Mapping of loss names to the corresponding loss functions and weights. The names will be used
                for logging purposes. The values should be dictionaries with the keys "loss" and "weight", the former
                containing the (nn.Module) loss function and the latter the (scalar) weight.
        """
        super().__init__()
        assert all("loss" in v and "weight" in v and len(v) == 2 for v in kwargs.values())
        self.loss_module_dict = nn.ModuleDict({k: v["loss"] for k, v in kwargs.items()})
        self.weight_dict = {k: v["weight"] for k, v in kwargs.items()}

    def forward(self, batch: OFData, **kwargs) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """The weighted sum of the energy loss, the gradient loss and the coefficient loss.

        Loss = energy_weight * energy_loss + gradient_weight * gradient_loss
        + coefficient_weight * coefficient_loss.

        Args:
            batch (Batch): Batch object containing the target energy and target gradients
            **kwargs: Additional arguments to be passed to the loss functions

        Returns:
            dict[str, Tensor]: Dictionary containing the weights for each loss component
            dict[str, Tensor]: Dictionary containing the losses for each loss component
        """

        loss_dict = {}
        for key, loss in self.loss_module_dict.items():
            loss_dict[key] = loss(batch, **kwargs)

        return self.weight_dict, loss_dict


class FullLoss(nn.Module):
    """Previous version of the WeightedLoss module, which is no longer supported.

    Included to make loading old checkpoints possible.
    """

    def __init__(self):
        """Just raises an error."""
        raise NotImplementedError("FullLoss is no longer supported, use WeightedLoss instead.)")
