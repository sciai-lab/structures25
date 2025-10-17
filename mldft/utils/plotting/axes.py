import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pyscf
from matplotlib import transforms
from numpy.typing import ArrayLike
from pyscf.lib.parameters import ANGULAR

from mldft.ml.data.components.basis_info import BasisInfo


def format_basis_func_xaxis(
    ax: plt.Axes,
    basis_info: BasisInfo,
    atom_types: None | ArrayLike = None,
) -> None:
    """Formats the x-axis of ``ax`` in-place such that the basis function indices are shown, and
    the atom types are indicated below.

    Args:
        ax: The axes object to format.
        basis_info: The basis info object.
        atom_types: The atomic numbers of the atoms to show on the x-axis.
            If None, each element in the basis is shown once.
    """

    if atom_types is None:  # show atom types as they appear in the basis
        atom_symbols = [
            pyscf.data.elements.ELEMENTS[atomic_number]
            for atomic_number in basis_info.atomic_numbers
        ]
        atom_types = basis_info.atomic_numbers
        basis_dim_per_atom = basis_info.basis_dim_per_atom

        l_per_shell = basis_info.l_per_shell
        shell_boundaries = basis_info.shell_to_first_basis_func
    else:  # show only the specified atom types (possibly multiple times)
        atom_types = np.asarray(
            [
                atomic_number
                for atomic_number in atom_types
                if atomic_number in basis_info.atomic_numbers
            ],
            dtype=np.int8,
        )
        atom_symbols = [
            pyscf.data.elements.ELEMENTS[atomic_number] for atomic_number in atom_types
        ]

        # indices of the atoms in the basis info object
        atom_ind = np.array(
            [basis_info.atomic_number_to_atom_index[atomic_number] for atomic_number in atom_types]
        )
        basis_dim_per_atom = basis_info.basis_dim_per_atom[atom_ind]

        l_per_shell = np.concatenate(
            np.asarray(
                np.split(basis_info.l_per_shell, np.cumsum(basis_info.n_shells_per_atom)[:-1]),
                dtype=object,
            )[atom_ind]
        )  # sorry

        shell_boundaries = np.concatenate([[0], np.cumsum(2 * l_per_shell + 1)])[:-1]

    n_basis = 0
    atom_indeces = []
    irreps_per_atom = []
    for atomic_number in atom_types:
        atom_indx = basis_info.atomic_number_to_atom_index[atomic_number]
        atom_indeces.append(atom_indx)
        n_basis += basis_info.basis_dim_per_atom[atom_indx]
        irreps_per_atom.append(basis_info.irreps_per_atom[atom_indx])

    ax.set_xlim(0, n_basis)

    shell_sizes = []  # number of coeffs per shell, i.e. always 2l+1
    l_sizes = []  # number of coeffs per l, e.g. 30 if there are 10 p functions
    l_symbols = []
    for irrep in irreps_per_atom:
        ls, counts = np.unique(irrep.ls, return_counts=True)
        shell_sizes.append(
            np.concatenate([np.full(count, 2 * L + 1) for L, count in zip(ls, counts)])
        )
        l_sizes.append((2 * ls + 1) * counts)
        l_symbols.extend([ANGULAR[l] for l in ls])

    xtick_positions = np.cumsum(np.concatenate([[0]] + shell_sizes))
    ax.set_xticks(
        xtick_positions,
        labels=[
            None,
        ]
        * len(xtick_positions),
        minor=True,
    )

    l_change_positions = np.cumsum(np.concatenate([[0]] + l_sizes))
    ax.set_xticks(
        l_change_positions,
        labels=[None] * len(l_change_positions),
        minor=False,
        horizontalalignment="center",
    )

    # Change the size of the x-tick marks
    # Convert from points to figure units (for consistent size regardless of plot size)
    ax_height = ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted()).height
    points_to_figure_units = plt.gcf().dpi_scale_trans.inverted().transform([1, 0])[0]
    ax.tick_params(axis="x", which="major", length=1700 * points_to_figure_units)

    l_center_positions = (l_change_positions[1:] + l_change_positions[:-1]) / 2
    # add text below the x-axis to indicate the l values using their symbols
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for center, symbol in zip(l_center_positions, l_symbols):
        ax.text(
            center,
            -12 * points_to_figure_units / ax_height,
            symbol,
            ha="center",
            va="center",
            transform=trans,
            fontsize="small",
        )

    # Second X-axis
    ax2 = ax.twiny()

    ax2.spines["bottom"].set_position(("axes", -24 * points_to_figure_units / ax_height))
    ax2.tick_params("both", length=0, width=0, which="minor")
    ax2.tick_params("both", direction="in", length=1700 * points_to_figure_units, which="major")
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")

    atom_boundaries = np.concatenate([[0], np.cumsum(basis_dim_per_atom)])

    # vertical lines at shell changes
    shell_boundaries = shell_boundaries[l_per_shell > 0]  # exclude s shells
    for boundary in shell_boundaries[1:-1]:
        ax.axvline(boundary, c="k", lw=0.75, ls="--", alpha=0.1)

    # vertical lines at l changes
    l_boundaries = np.concatenate([[0], l_change_positions])
    for boundary in l_boundaries[1:-1]:
        ax.axvline(boundary, c="k", lw=0.75, ls="--", alpha=0.3)

    # vertical lines at atom boundaries
    for boundary in atom_boundaries[1:-1]:
        ax.axvline(boundary, c="k", lw=0.75, ls="-")

    ax2.set_xticks(l_change_positions)
    ax2.xaxis.set_major_formatter(ticker.NullFormatter())
    ax2.xaxis.set_minor_locator(
        ticker.FixedLocator((atom_boundaries[1:] + atom_boundaries[:-1]) / 2)
    )
    ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(atom_symbols))
    ax2.set_xlabel("basis function index")
