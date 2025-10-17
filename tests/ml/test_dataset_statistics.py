import os

import numpy as np
import pytest

from mldft.ml.models.components.sample_weighers import ConstantSampleWeigher
from mldft.ml.preprocess.dataset_statistics import DatasetStatistics


@pytest.mark.parametrize("with_bias", [True, False])
def test_DatasetStatistics_from_dataset(
    dummy_dataset_torch, dummy_basis_info, with_bias, tmp_path
):
    """Test the fit of dataset statistics.

    Only the shapes are checked.
    """
    statistics = DatasetStatistics.from_dataset(
        path=tmp_path / "dummy_statistics.zarr",
        dataset=dummy_dataset_torch,
        basis_info=dummy_basis_info,
        with_bias=with_bias,
        atom_ref_fit_sample_weighers=[ConstantSampleWeigher()],
        initial_guess_sample_weighers=[ConstantSampleWeigher()],
    )

    assert statistics.load_statistic("constant", "coeffs/mean").shape == (
        dummy_basis_info.n_basis,
    )
    assert statistics.load_statistic("constant", "coeffs/std").shape == (dummy_basis_info.n_basis,)
    assert statistics.load_statistic("constant", "gradient_label/abs_max").shape == (
        dummy_basis_info.n_basis,
    )
    assert statistics.load_statistic("constant", "atom_ref_atom_type_bias").shape == (
        dummy_basis_info.n_types,
    )
    assert statistics.load_statistic("constant", "atom_ref_global_bias").shape == ()


def test_DatasetStatistics_save_to(dummy_dataset_statistics, tmp_path_factory):
    """Test saving and loading of DatasetStatistics."""
    path = tmp_path_factory.getbasetemp().joinpath("statistics.zarr")
    print(f"Saving to {path}")
    dummy_dataset_statistics.save_to(path)
    statistics_loaded = DatasetStatistics(path)
    for key in dummy_dataset_statistics.keys_recursive():
        weigher_key, statistics_key = key.split("/", 1)
        assert np.allclose(
            dummy_dataset_statistics.load_statistic(weigher_key, statistics_key),
            statistics_loaded.load_statistic(weigher_key, statistics_key),
        ), f"Field {key} mismatch."

    # saving to the same path again should work, as all the statistics match
    dummy_dataset_statistics.save_to(path)

    with pytest.raises(FileNotFoundError):
        DatasetStatistics(os.path.join(path, "non_existent_file.zarr"))
