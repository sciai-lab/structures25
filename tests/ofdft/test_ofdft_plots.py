"""Tests to easily debug ofdft plots."""
import pytest
import torch
from matplotlib import pyplot as plt

from mldft.utils.plotting.summary_density_optimization import plot_quantiles_data


@pytest.mark.skip()
def test_quantile_plots():
    """Test the quantile plots."""
    torch.manual_seed(42)
    num_datasets = 100
    max_datasets = 50
    data_y = torch.normal(mean=10, std=2, size=(num_datasets, max_datasets))

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    # Apply the plot to the axes
    plot_quantiles_data(data_y, ax)

    plt.tight_layout()
    plt.show()
