import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection


def comparison_scatter(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray, **scatter_kwargs
) -> PathCollection:
    """Scatter plot with diagonal line behind it. Shuffles the data to avoid overplotting.

    Args:
        ax (plt.Axes): Axes object to plot on
        x (np.ndarray): x values
        y (np.ndarray): y values
        **scatter_kwargs: Keyword arguments for the scatter plot. Defaults to marker='.', s=1.

    Returns:
        PathCollection: The PathCollection of the scatter plot.
    """

    scatter_kwargs = scatter_kwargs.copy()
    scatter_kwargs.setdefault("marker", ".")
    scatter_kwargs.setdefault("s", 1)

    # shuffle the data to avoid overplotting
    shuffle = np.random.permutation(len(x))
    x = x[shuffle]
    y = y[shuffle]
    for key in ["c", "marker", "alpha", "s"]:
        if key in scatter_kwargs and isinstance(scatter_kwargs[key], np.ndarray):
            scatter_kwargs[key] = scatter_kwargs[key][shuffle]

    scatter = ax.scatter(x, y, **scatter_kwargs)

    # make the limits of x and y axis equal
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    lim = [min(x_min, y_min), max(x_max, y_max)]
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    # draw diagonal line behind scatter plot
    ax.plot(lim, lim, color="k", alpha=1, zorder=-1, linewidth=0.5)

    ax.set_aspect("equal")
    return scatter
