from matplotlib import pyplot as plt


def set_equal_aspect_and_square_limits(ax: plt.Axes) -> None:
    """Set the limits of an axes object to be square and equal aspect ratio (Using only
    ``ax.set_aspect('equal')`` will result in a plot that is not square).

    Args:
        ax: The axes object to modify.
    """

    # set the limits to make the plot square
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    # set the range as the maximum of x and y ranges
    ax_range = max(x_max - x_min, y_max - y_min)
    # keep the points centered, but extend the limits as needed
    x_padding = (ax_range - (x_max - x_min)) / 2
    x_min, x_max = x_min - x_padding, x_max + x_padding
    y_padding = (ax_range - (y_max - y_min)) / 2
    y_min, y_max = y_min - y_padding, y_max + y_padding

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
