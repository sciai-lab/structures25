import os
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from mldft.ml.data.components.convert_transforms import AddOverlapMatrix
from mldft.ml.models.components.natural_reparametrization import (
    natural_reparametrization_matrices,
)
from mldft.utils.utils import set_default_torch_dtype

TEST_DATASET = "QM9subset"
TEST_DATA_DIR_EXISTS = (Path(os.getenv("DFT_DATA")) / TEST_DATASET).exists()


@set_default_torch_dtype(torch.float64)
@pytest.fixture
@pytest.mark.skipif(not TEST_DATA_DIR_EXISTS, reason=f"Test dataset {TEST_DATASET} not available.")
def real_dummy_sample_and_basis_info():
    """Create a dummy sample and basis_info from the test dataset."""

    with initialize(
        version_base=None,
        config_path="../../configs/ml/",
        job_name="natural_reparametrization",
    ):
        cfg = compose(config_name="train.yaml", overrides=[f"data={TEST_DATASET.lower()}"])

    datamodule = instantiate(cfg.data.datamodule)
    datamodule.setup("fit")

    real_dummy_sample = datamodule.train_set[0]  # fixed sample for now
    real_dummy_basis_info = datamodule.train_set.basis_info

    add_overlap_matrix = AddOverlapMatrix(real_dummy_basis_info)
    real_dummy_sample = add_overlap_matrix(real_dummy_sample)

    return real_dummy_sample, real_dummy_basis_info


@pytest.mark.parametrize(
    "overlap_matrix",
    [
        np.array([[1, 0.5], [0.5, 1]]),
        np.array([[[1, 0.5], [0.5, 1]], [[2, 0.3], [0.3, 2]]]),
    ],
)
@set_default_torch_dtype(torch.float64)
@pytest.mark.parametrize("orthogonalization", ["symmetric", "canonical"])
def test_natural_reparametrization_matrices(overlap_matrix, orthogonalization):
    """Test that the matrices M and M^-1 are inverses of each other and that M^T M is equal to the
    overlap matrix / matrices for the pre-defined overlap matrix."""

    m, m_inv = natural_reparametrization_matrices(overlap_matrix)
    n = overlap_matrix.shape[-1]
    eyes = np.broadcast_to(np.eye(n), overlap_matrix.shape)
    np.testing.assert_allclose(m @ m_inv, eyes, rtol=1e-05, atol=1e-07)
    np.testing.assert_allclose(m_inv @ m, eyes, rtol=1e-05, atol=1e-07)
    np.testing.assert_allclose(m @ m.T, overlap_matrix, rtol=1e05, atol=1e-07)
    if orthogonalization == "symmetric":
        np.testing.assert_allclose(m @ m, overlap_matrix, rtol=1e-05, atol=1e-07)


@pytest.mark.skipif(not TEST_DATA_DIR_EXISTS, reason=f"Test dataset {TEST_DATASET} not available.")
@set_default_torch_dtype(torch.float64)
@pytest.mark.parametrize("orthogonalization", ["symmetric", "canonical"])
def test_symmetric_natural_reparametrization_matrices(
    real_dummy_sample_and_basis_info, orthogonalization
):
    """Test that the matrices M and M^-1 are inverses of each other and that M^T M is equal to the
    overlap matrix / matrices as well as M = M^T in the symmetric case for actual data samples."""

    real_dummy_sample, _ = real_dummy_sample_and_basis_info
    overlap_matrix = real_dummy_sample.overlap_matrix

    m, m_inv = natural_reparametrization_matrices(
        overlap_matrix, orthogonalization=orthogonalization
    )
    n = overlap_matrix.shape[-1]
    eyes = np.broadcast_to(np.eye(n), overlap_matrix.shape)
    np.testing.assert_allclose(m @ m_inv, eyes, rtol=1e-05, atol=1e-07)
    np.testing.assert_allclose(m_inv @ m, eyes, rtol=1e-05, atol=1e-07)
    np.testing.assert_allclose(m @ m.T, overlap_matrix, rtol=1e05, atol=1e-07)
    if orthogonalization == "symmetric":
        np.testing.assert_allclose(m, m.T, rtol=1e-05, atol=1e-07)
        np.testing.assert_allclose(m_inv, m_inv.T, rtol=1e-05, atol=1e-07)
        np.testing.assert_allclose(m @ m, overlap_matrix, rtol=1e-05, atol=1e-07)
