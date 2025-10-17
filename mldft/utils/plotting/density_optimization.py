from pathlib import Path

import numpy as np
import torch
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from pyscf import dft

from mldft.ml.data.components.basis_transforms import transform_tensor_with_sample
from mldft.ml.data.components.of_data import OFData, Representation
from mldft.ofdft.callbacks import BasicCallback
from mldft.ofdft.energies import Energies


def plot_density_optimization(
    callback: BasicCallback,
    energies_label: Energies,
    coeffs_label: torch.Tensor,
    sample: OFData,
    stopping_index: int = None,
    basis_l: torch.Tensor = None,
    figure_path: Path | str = None,
    enable_grid_operations: bool = False,
):
    """Plot the density optimization.

    If a figure path is given, the figure is saved to that path.

    Args:
        callback: The callback object.
        energies_label: The label energies.
        coeffs_label: The label coefficients.
        sample: The sample used for transformations.
        stopping_index: The index at which the optimization stopped. Optional.
        basis_l: Angular momentum of basis functions. Optional.
        figure_path: The path to save the figure. If None, the figure is not saved.
        enable_grid_operations: If False, plots requiring computations on the grid
            (the integrated negative density plot) are not shown.
    """

    plots = [
        _add_all_energy_differences,
        # _add_density_difference,
        # _add_gradient_norm,
        _add_density_and_gradient,
        # _add_coefficient_differences_pixels,
        _add_coefficient_differences_lines,
    ]
    if enable_grid_operations:
        plots.insert(2, _add_integrated_negative_density)

    fig, axes = plt.subplots(len(plots), 1, figsize=(9, 4 * len(plots)), sharex=False)

    coeffs_label = coeffs_label.detach()
    for plot, ax in zip(plots, axes):
        plot(
            ax=ax,
            callback=callback,
            energies_label=energies_label,
            coeffs_label=coeffs_label,
            basis_l=basis_l,
            stopping_index=stopping_index,
            sample=sample,
        )

    plt.tight_layout()

    if figure_path is not None:
        plt.savefig(figure_path)


def _add_density_difference(
    ax: plt.Axes,
    callback: BasicCallback,
    energies_label: Energies,
    coeffs_label: torch.Tensor,
    sample: OFData,
    stopping_index: int = None,
    **kwargs,
):
    """Add the density difference to the given axis.

    Args:
        ax: The axis to add the density difference to.
        callback: The callback object.
        energies_label: The label energies.
        coeffs_label: The label coefficients.
        stopping_index: The index at which the optimization stopped. Optional.
        **kwargs: Additional arguments for uniform plot interface.
    """
    l2_norm = callback.l2_norm
    ax.plot(l2_norm, label="coeffs delta")
    ax.set_ylabel(r"$\Vert\rho_\mathrm{pred} - \rho_\mathrm{target}\Vert_2$ [electrons]")
    ax.set_yscale("log")
    ax.grid()
    title = "Density difference"
    if stopping_index is not None:
        ax.axvline(stopping_index, color="black", linestyle="--")
        title += f". Error at stopping index: {l2_norm[stopping_index]:.2g} electrons"
    ax.set_title(title)
    ax.set_xlabel("Iteration")


def _add_all_energy_differences(
    ax: plt.Axes,
    callback: BasicCallback,
    energies_label: Energies,
    stopping_index: int = None,
    **kwargs,
):
    """Add all energy differences to the given axis.

    Args:
        ax: The axis to add the energy differences to.
        callback: The callback object.
        energies_label: The label energies.
        stopping_index: The index at which the optimization stopped. Optional.
        **kwargs: Additional arguments for uniform plot interface.
    """
    linthresh = 10

    total_energy = torch.as_tensor([e.total_energy for e in callback.energy], dtype=torch.float64)
    ax.plot(1e3 * (total_energy - energies_label.total_energy), label=r"$\Delta E_\mathrm{tot}$")

    for energy_name in energies_label.energies_dict:
        if energy_name == "nuclear_repulsion":
            continue
        try:
            energy_np = torch.as_tensor([e[energy_name] for e in callback.energy])
        except KeyError:
            logger.warning(f"Energy '{energy_name}' not found in callback")
            continue
        label = r"$\Delta E_\mathrm{" + energy_name.replace("_", r"\_") + "}$"
        ax.plot(1e3 * (energy_np - energies_label[energy_name]), label=label)

    ax.axhline(0, color="black")
    if stopping_index is not None:
        ax.axvline(stopping_index, color="black", linestyle="--", label="stopping index")

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\Delta E$ [mHa]")
    ax.set_yscale("symlog", linthresh=linthresh)
    title = "Energy differences"
    if stopping_index is not None:
        title += (
            f". Error at stopping index: "
            f"{1e3 * (total_energy[stopping_index] - energies_label.total_energy):.2f} mHa"
        )
    ax.set_title(title)
    ax.legend()
    ax.grid(which="both")
    for h in torch.linspace(-linthresh, linthresh, 21)[1:-1]:
        ax.axhline(h, color="black", alpha=0.1, linestyle="--", linewidth=0.5)


def _add_density_and_gradient(
    ax: plt.Axes,
    callback: BasicCallback,
    energies_label: Energies,
    coeffs_label: torch.Tensor,
    sample: OFData,
    stopping_index: int = None,
    **kwargs,
):
    """Add the density and gradient to the given axis as a pixel image.

    Args:
        ax: The axis to add the density and gradient to.
        callback: The callback object.
        energies_label: The label energies used to obtain the overlap matrix via the mol.
        coeffs_label: The label coefficients.
        sample: The sample used for transformations.
        stopping_index: The index at which the optimization stopped. Optional.
        **kwargs: Additional arguments for uniform plot interface.
    """
    _add_density_difference(
        ax,
        callback=callback,
        energies_label=energies_label,
        coeffs_label=coeffs_label,
        sample=sample,
        stopping_index=stopping_index,
    )
    twin_ax = plt.twinx(ax)
    _add_gradient_norm(twin_ax, callback, stopping_index)
    twin_ax.grid(False)  # don't show grid twice

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = twin_ax.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center")
    ax.set_xlabel("Iteration")


def _add_gradient_norm(
    ax: plt.Axes,
    callback: BasicCallback,
    stopping_index: int = None,
    **kwargs,
):
    """Add the gradient norm to the given axis.

    Args:
        ax: The axis to add the gradient norm to.
        callback: The callback object.
        stopping_index: The index at which the optimization stopped. Optional.
        **kwargs: Additional arguments for uniform plot interface.
    """

    ax.plot(callback.gradient_norm, label="gradient norm", color="tab:orange")
    ax.set_ylabel("gradient norm")
    ax.set_xlabel("Iteration")
    ax.set_yscale("log")
    ax.grid()
    if stopping_index is not None:
        ax.axvline(stopping_index, color="black", linestyle="--")


def _add_coefficient_differences_pixels(
    ax: plt.Axes,
    callback: BasicCallback,
    coeffs_label: torch.Tensor,
    **kwargs,
):
    """Add the coefficient differences to the given axis as a pixel image.

    Args:
        ax: The axis to add the coefficient difference to.
        callback: The callback object.
        coeffs_label: The label coefficients.
        **kwargs: Additional arguments for uniform plot interface.
    """

    coeffs = torch.stack(callback.coeffs)
    coeffs_delta = (coeffs - coeffs_label).cpu()
    coeffs_delta_max = torch.max(torch.abs(coeffs_delta)).item()
    img = ax.imshow(
        coeffs_delta.T.detach().numpy(),
        aspect="auto",
        cmap="seismic",
        interpolation="none",
        vmin=-coeffs_delta_max,
        vmax=coeffs_delta_max,
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"basis function")
    plt.colorbar(img, ax=ax, location="top")


def _add_coefficient_differences_lines(
    ax: plt.Axes,
    callback: BasicCallback,
    coeffs_label: torch.Tensor,
    basis_l: torch.Tensor = None,
    stopping_index: int = None,
    **kwargs,
):
    """Add the coefficient differences to the given axis as a plot.

    Args:
        ax: The axis to add the coefficient difference to.
        callback: The callback object.
        coeffs_label: The label coefficients.
        basis_l: Angular momentum of basis functions.
        stopping_index: The index at which the optimization stopped. Optional.
        **kwargs: Additional arguments for uniform plot interface.
    """

    coeffs = torch.stack(callback.coeffs).to(coeffs_label.device)
    coeffs_delta = (coeffs - coeffs_label).cpu()

    if basis_l is None:
        ax.plot(coeffs_delta.detach().numpy(), lw=0.3)
    else:
        max_l = np.max(basis_l)
        orbital_labels = np.array(["s", "p", "d", "f", "g", "h"])
        l_colors = np.array(
            ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
        )
        l_colors_handles = [
            Patch(facecolor=color, label=label)
            for color, label in zip(l_colors[: max_l + 1], orbital_labels)
        ]
        ax.legend(handles=l_colors_handles, title="Angular Momentum", fontsize="small")

        # shuffle z order
        np.random.seed(0)
        permutation = np.random.permutation(basis_l.shape[0])
        for i in permutation:
            ax.plot(coeffs_delta[:, i].detach().numpy(), lw=0.3, c=l_colors[basis_l[i]])

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\Delta p$")
    ax.grid()

    if stopping_index is not None:
        ax.axvline(stopping_index, color="black", linestyle="--")


def _add_integrated_negative_density(
    ax: plt.Axes,
    callback: BasicCallback,
    energies_label: Energies,
    sample: OFData,
    stopping_index: int = None,
    **kwargs,
):
    """Add the integrated negative density to the given axis.

    Args:
        ax: The axis to add the integrated negative density to.
        energies_label: The label energies.
        sample: The sample used for transformations.
        stopping_index: The index at which the optimization stopped. Optional.
        **kwargs: Additional arguments for uniform plot interface.
    """
    mol = sample.mol
    coeffs = torch.stack(callback.coeffs)

    grid = dft.Grids(mol)
    grid.level = 3
    grid.prune = None
    grid.build()

    weights = torch.as_tensor(grid.weights, dtype=torch.float64)
    # weights = torch.clip(weights, min=0, max=np.inf)
    ao = dft.numint.eval_ao(mol, grid.coords, deriv=0)
    ao = torch.as_tensor(ao, dtype=torch.float64, device=sample.coeffs.device)
    ao = transform_tensor_with_sample(sample, ao, Representation.AO)
    ao = ao.to("cpu")

    density = coeffs @ ao.T
    density[density > 0] = 0
    integrated_negative_density = density @ weights

    ax.plot(integrated_negative_density, label="integrated negative density")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("int. neg. density [electrons]")
    ax.grid()

    if stopping_index is not None:
        ax.axvline(stopping_index, color="black", linestyle="--")
