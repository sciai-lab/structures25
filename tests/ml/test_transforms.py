import copy
import os
from pathlib import Path

import pytest
import torch.testing
from hydra import compose, initialize
from hydra.utils import instantiate

from mldft.ml.data.components.basis_transforms import (
    MasterTransformation,
    ToGlobalNatRep,
    ToLocalFrames,
)
from mldft.ml.data.components.convert_transforms import (
    AddAtomCooIndices,
    AddOverlapMatrix,
    ToTorch,
)
from mldft.utils.utils import set_default_torch_dtype

TEST_DATASET = "QM9subset"
TEST_DATA_DIR_EXISTS = (Path(os.getenv("DFT_DATA")) / TEST_DATASET).exists()


@set_default_torch_dtype(torch.float64)
@pytest.fixture
@pytest.mark.skipif(not TEST_DATA_DIR_EXISTS, reason=f"Test dataset {TEST_DATASET} not available.")
def untransformed_sample_and_basis_info():
    """Create a dummy sample and basis_info from the test dataset."""
    with initialize(
        version_base=None,
        config_path="../../configs/ml/",
        job_name="natural_reparametrization",
    ):
        cfg = compose(
            config_name="train.yaml",
            overrides=["data=qm9subset", "data/transforms=none"],
        )

    datamodule = instantiate(cfg.data.datamodule)
    datamodule.setup("fit")

    real_dummy_sample = datamodule.train_set[0]  # fixed sample for now
    real_dummy_basis_info = datamodule.train_set.basis_info

    return copy.deepcopy(real_dummy_sample), copy.deepcopy(real_dummy_basis_info)


@set_default_torch_dtype(torch.float64)
@pytest.mark.parametrize("do_local_frames", [False, True])
@pytest.mark.parametrize("orthogonalization", ["symmetric", "canonical"])
@pytest.mark.skipif(not TEST_DATA_DIR_EXISTS, reason=f"Test dataset {TEST_DATASET} not available.")
def test_global_natural_reparametrization_transform(
    untransformed_sample_and_basis_info, do_local_frames, orthogonalization
):
    """Test the global natural reparametrization transform."""
    sample, basis_info = untransformed_sample_and_basis_info
    add_overlap_matrix = AddOverlapMatrix(basis_info)
    to_torch = ToTorch()
    sample = add_overlap_matrix(sample)
    sample = to_torch(sample)
    if do_local_frames:
        sample = AddAtomCooIndices()(sample)
        sample = ToLocalFrames()(sample)
    natrep = ToGlobalNatRep(orthogonalization=orthogonalization)
    transformed_sample = natrep(sample.clone())
    density_change = sample.coeffs @ sample.overlap_matrix @ sample.coeffs
    density_change_transformed = transformed_sample.coeffs @ transformed_sample.coeffs
    torch.testing.assert_close(
        transformed_sample.overlap_matrix,
        torch.eye(transformed_sample.overlap_matrix.shape[-1]),
    )
    torch.testing.assert_close(density_change, density_change_transformed)


@set_default_torch_dtype(torch.float64)
def test_master_transform(dummy_sample, dummy_basis_info):
    """Test the master transform."""
    master_transforms = [
        MasterTransformation(
            name="no_basis_transforms",
            pre_transforms=[ToTorch(float_dtype=torch.float64)],
        ),
        MasterTransformation(
            name="local_frames",
            pre_transforms=[ToTorch(float_dtype=torch.float64)],
            basis_transforms=[ToLocalFrames(sparse=False)],
        ),
        MasterTransformation(
            name="global_natrep",
            pre_transforms=[
                AddOverlapMatrix(dummy_basis_info),
                ToTorch(float_dtype=torch.float64),
            ],
            basis_transforms=[ToGlobalNatRep()],
        ),
        MasterTransformation(
            name="local_frames_global_natrep",
            pre_transforms=[
                AddOverlapMatrix(dummy_basis_info),
                ToTorch(float_dtype=torch.float64),
            ],
            basis_transforms=[ToLocalFrames(sparse=False), ToGlobalNatRep()],
        ),
    ]
    for master_transform in master_transforms:
        transformed_sample = master_transform(dummy_sample.clone())
        retransformed_sample = master_transform.invert_basis_transform(transformed_sample.clone())
        torch.testing.assert_close(
            torch.as_tensor(dummy_sample.coeffs, dtype=torch.float64),
            retransformed_sample.coeffs,
        )
        torch.testing.assert_close(
            torch.as_tensor(dummy_sample.gradient_label, dtype=torch.float64),
            retransformed_sample.gradient_label,
        )
