import logging
from typing import Any

import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from pyscf.data.elements import ELEMENTS

from mldft.ml.callbacks.base import OnStepCallbackWithTiming
from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.models.mldft_module import MLDFTLitModule
from mldft.utils.plotting.axes import format_basis_func_xaxis
from mldft.utils.plotting.scatter import comparison_scatter

logger = logging.getLogger(__name__)


class LogMatplotlibToTensorboard(OnStepCallbackWithTiming):
    """Base Class to log matplotlib figures to tensorboard."""

    def get_figure(
        self,
        pl_module: MLDFTLitModule,
        batch: Any,
        outputs: STEP_OUTPUT,
        basis_info: BasisInfo,
    ) -> Figure:
        """Create the figure to be plotted.

        Args:
            pl_module: The lightning module.
            batch: The batch.
            outputs: The outputs of the lightning module.
            basis_info: The :class:`BasisInfo` object.

        Returns:
            Figure: The figure to be plotted.
        """
        raise NotImplementedError

    def execute(
        self,
        trainer: pl.Trainer,
        pl_module: MLDFTLitModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        split: str,
    ) -> None:
        """Generates a figure using :meth:`get_figure` and logs it to tensorboard."""

        logger.debug(f"Logging training figure to tensorboard from {self.__class__.__name__}")

        basis_info = pl_module.basis_info

        tb_logger = pl_module.tensorboard_logger
        tb_logger.add_figure(
            f"{split}_{self.name}",
            self.get_figure(pl_module, batch, outputs, basis_info),
            global_step=trainer.global_step,
        )


class LogTargetPredScatters(LogMatplotlibToTensorboard):
    """Logs scatter plots of the target and predicted energies, gradients and initial guess
    deltas."""

    def __init__(self, with_atom_ref: bool | str = "auto", **super_kwargs):
        """
        Args:
            with_atom_ref (bool | str): Whether to add two additional plots of energy / gradient minus their AtomRef
                values. If 'auto', it is checked whether the model has a property ``atom_ref_module``.
                Defaults to 'auto'.
            **super_kwargs: Additional kwargs for the superclass.
        """
        super().__init__(**super_kwargs)
        assert with_atom_ref in [
            True,
            False,
            "auto",
        ], f'with_atom_ref must be one of [True, False, "auto"], got {with_atom_ref}'
        self.with_atom_ref = with_atom_ref

    def get_figure(
        self,
        pl_module: MLDFTLitModule,
        batch: Any,
        outputs: STEP_OUTPUT,
        basis_info: BasisInfo,
    ) -> Figure:
        """Create two scatter plots: One for the energies, and one for the gradients.

        Args:
            pl_module: The lightning module.
            batch: The batch.
            outputs: The outputs of the lightning module.
            basis_info: The :class:`BasisInfo` object.

        Returns:
            Figure: The figure to be plotted.
        """
        # mask out samples without energy labels for energy and gradient plots
        mask = batch.has_energy_label.detach().cpu().numpy()

        # gradients
        gradient_label = batch.gradient_label.detach().cpu().numpy()
        # pred_gradients = outputs["model_outputs"]["pred_gradients"].detach().cpu().numpy()
        projected_gradient_difference = (
            outputs["projected_gradient_difference"].detach().cpu().numpy()
        )

        # energies
        energy_labels = batch.energy_label.detach().cpu().numpy()
        pred_energies = outputs["model_outputs"]["pred_energy"].detach().cpu().float().numpy()

        # initial guess delta
        pred_diff = outputs["model_outputs"]["pred_diff"].detach().cpu().float().numpy()
        gt_diff = (batch.coeffs - batch.ground_state_coeffs).detach().cpu().float().numpy()

        # prepare to color by scf iteration
        scf_iteration = batch.scf_iteration.cpu().numpy()
        scf_iteration_per_basis_func = scf_iteration[batch.coeffs_batch.cpu().numpy()]

        n_scf_iteration_colors = 7  # iterations 0-5, 6+ is the same color
        scf_iteration_colors = np.array([f"C{i}" for i in range(n_scf_iteration_colors)])
        scf_iteration_colors_per_coeff = scf_iteration_colors[
            np.clip(scf_iteration_per_basis_func, 0, len(scf_iteration_colors) - 1)
        ]
        scf_iteration_handles = [
            Patch(
                facecolor=color,
                label=i if i < len(scf_iteration_colors) - 1 else f">{i-1}",
            )
            for i, color in enumerate(scf_iteration_colors)
        ]
        scf_iteration_legend_kwargs = dict(
            handles=scf_iteration_handles,
            title="SCF Iteration",
            loc="lower right",
            fontsize="small",
        )

        # prepare to color by atom type
        atom_ind = batch.atom_ind.cpu().numpy()
        atom_ind_per_coeff = atom_ind[batch.coeff_ind_to_node_ind.cpu().numpy().flatten()]
        atom_ind_colors = np.array([f"C{str(i).zfill(2)}" for i in range(basis_info.n_types)])
        atom_ind_colors_per_coeff = atom_ind_colors[atom_ind_per_coeff]
        atom_ind_handles = [
            Patch(facecolor=color, label=ELEMENTS[atomic_number])
            for color, atomic_number in zip(atom_ind_colors, basis_info.atomic_numbers)
        ]
        atom_ind_legend_kwargs = dict(
            handles=atom_ind_handles,
            title="Element",
            loc="lower right",
            fontsize="small",
        )

        # prepare to color by angular momentum
        basis_func_to_l = basis_info.l_per_basis_func[batch.basis_function_ind.cpu().numpy()]
        max_l = np.max(basis_func_to_l)
        orbital_labels = np.array(["s", "p", "d", "f", "g", "h"])
        l_colors = np.array(
            [
                "tab:blue",
                "tab:orange",
                "tab:green",
                "tab:red",
                "tab:purple",
                "tab:brown",
            ]
        )
        l_colors_per_coeff = l_colors[basis_func_to_l]
        l_colors_handles = [
            Patch(facecolor=color, label=label)
            for color, label in zip(l_colors[: max_l + 1], orbital_labels)
        ]
        l_color_legend_kwargs = dict(
            handles=l_colors_handles,
            title="Angular Momentum",
            loc="lower right",
            fontsize="small",
        )

        if self.with_atom_ref == "auto":
            with_atom_ref = hasattr(pl_module.net, "atom_ref_module")
        else:
            with_atom_ref = self.with_atom_ref
        if with_atom_ref:
            try:
                atom_ref = pl_module.net.atom_ref_module
            except Exception:
                raise ValueError(
                    "Could not get the atom_ref_module, but with_atom_ref is set to True."
                )
            with torch.enable_grad():
                # batch.coeffs.requires_grad_()
                pred_energies_atom_ref = atom_ref.sample_forward(batch)
                pred_gradient_atom_ref = (
                    torch.autograd.grad(pred_energies_atom_ref.sum(), batch.coeffs)[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                pred_energies_atom_ref = pred_energies_atom_ref.detach().cpu().numpy()
        else:
            pred_energies_atom_ref = None
            pred_gradient_atom_ref = None

        n_rows = 3
        n_cols = 4 if with_atom_ref else 3
        plot_size = 6
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * plot_size, n_rows * plot_size))

        ax = axs[0, 0]
        comparison_scatter(
            ax,
            energy_labels[mask],
            pred_energies[mask],
            c=scf_iteration_colors[np.clip(scf_iteration, 0, len(scf_iteration_colors) - 1)][mask],
            s=12,
        )
        ax.set_title("Target vs. Predicted Energies")
        ax.set_xlabel("Target Energy")
        ax.set_ylabel("Predicted Energy")
        ax.legend(**scf_iteration_legend_kwargs)

        if with_atom_ref:
            ax = axs[1, 0]
            comparison_scatter(
                ax,
                (energy_labels - pred_energies_atom_ref)[mask],
                (pred_energies - pred_energies_atom_ref)[mask],
                c=scf_iteration_colors[np.clip(scf_iteration, 0, len(scf_iteration_colors) - 1)][
                    mask
                ],
                s=12,
            )
            ax.set_title("Target vs. Predicted Energies minus AtomRef")
            ax.set_xlabel("Target Energy minus AtomRef")
            ax.set_ylabel("Predicted Energy minus AtomRef")
            ax.legend(**scf_iteration_legend_kwargs)

        mask_per_coeff = mask[batch.coeffs_batch.cpu().numpy()]

        for row, c, legend_kwargs in zip(
            [0, 1, 2],
            [
                scf_iteration_colors_per_coeff,
                l_colors_per_coeff,
                atom_ind_colors_per_coeff,
            ],
            [
                scf_iteration_legend_kwargs,
                l_color_legend_kwargs,
                atom_ind_legend_kwargs,
            ],
        ):
            ax = axs[row, 1]
            comparison_scatter(
                ax,
                gradient_label[mask_per_coeff],
                (gradient_label + projected_gradient_difference)[mask_per_coeff],
                c=c[mask_per_coeff],
            )
            ax.set_title("Target Gradients vs.\nTarget Gradients + Projected gradient difference")
            ax.set_xlabel("Target Gradient")
            ax.set_ylabel("Predicted Gradient")
            ax.legend(**legend_kwargs)

            if with_atom_ref:
                ax = axs[row, 2]
                comparison_scatter(
                    ax,
                    (gradient_label - pred_gradient_atom_ref)[mask_per_coeff],
                    (gradient_label + projected_gradient_difference - pred_gradient_atom_ref)[
                        mask_per_coeff
                    ],
                    c=c[mask_per_coeff],
                )
                ax.set_title(
                    "Target Gradients minus AtomRef vs.\n"
                    "Target Gradients + Projected gradient difference - AtomRef"
                )
                ax.set_xlabel("Target Gradient minus AtomRef")
                ax.set_ylabel("Predicted Gradient minus AtomRef")
                ax.legend(**legend_kwargs)

            ax = axs[row, 3 if with_atom_ref else 2]
            comparison_scatter(
                ax,
                gt_diff,
                pred_diff,
                c=c,
            )
            ax.set_title("Target vs. Predicted Initial Guess Delta")
            ax.set_xlabel("Target Initial Guess Delta")
            ax.set_ylabel("Predicted Initial Guess Delta")
            ax.legend(**legend_kwargs)

        fig.tight_layout()

        # hide for now
        axs[2, 0].set_visible(False)
        if not with_atom_ref:
            axs[1, 0].set_visible(False)

        return fig


class LogGradientScatter(LogMatplotlibToTensorboard):
    """Logs a scatter plot of the target and predicted gradients per basis function."""

    def get_figure(
        self,
        pl_module: MLDFTLitModule,
        batch: Any,
        outputs: STEP_OUTPUT,
        basis_info: BasisInfo,
    ) -> Figure:
        """Create the figure to be plotted: A scatter plot of the target and predicted gradients per
        basis function, as well as their projected difference.

        Args:
            pl_module: The lightning module.
            batch: The batch.
            outputs: The outputs of the lightning module.
            basis_info: The :class:`BasisInfo` object.

        Returns:
            Figure: The figure to be plotted.
        """
        gradient_label = batch.gradient_label.detach().cpu().numpy()
        pred_gradients = outputs["model_outputs"]["pred_gradients"].detach().cpu().numpy()
        projected_gradient_difference = (
            outputs["projected_gradient_difference"].detach().cpu().numpy()
        )
        fig, ax = plt.subplots(1, figsize=(15, 5))

        scatter_kwargs = dict(marker=".", alpha=0.1, s=1)
        x = batch.basis_function_ind.cpu().numpy() + 0.5
        ax.scatter(x, gradient_label, **scatter_kwargs, label="True Gradient")
        ax.scatter(x, pred_gradients, **scatter_kwargs, label="Predicted Gradient")
        ax.scatter(
            x,
            projected_gradient_difference,
            **scatter_kwargs,
            label="Projected Difference",
        )

        handles = [
            Patch(color="C0", label="True Gradient"),
            Patch(color="C1", label="Predicted Gradient"),
            Patch(color="C2", label="Projected Difference"),
        ]
        ax.legend(handles=handles, loc="upper right")

        format_basis_func_xaxis(ax, basis_info)
        fig.tight_layout()

        return fig


class LogDistanceEmbeddings(LogMatplotlibToTensorboard):
    """Logs a line plot of the distance embeddings for a range of distances."""

    def __init__(self, max_distance: float = 5.0, n_distances: int = 1000, **super_kwargs):
        """Plot the distance embeddings for a range of distances.

        Args:
            max_distance (float): The maximum distance to consider.
            n_distances (int): The number of distances to consider.

        Returns:
            plt.Figure: The plot.
        """
        super().__init__(**super_kwargs)
        self.max_distance = max_distance
        self.n_distances = n_distances

    def get_figure(
        self,
        pl_module: MLDFTLitModule,
        batch: Any,
        outputs: STEP_OUTPUT,
        basis_info: BasisInfo,
    ) -> Figure:
        """Create the figure to be plotted: A line plot of the distance embeddings"""
        if hasattr(pl_module.net, "plot_distance_embeddings"):
            return pl_module.net.plot_distance_embeddings()
        else:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.set_title("Module has no method plot_distance_embeddings.")
            return fig
