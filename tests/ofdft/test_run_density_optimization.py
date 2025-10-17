import pytest
import rootutils
import torch
from hydra import compose, initialize
from omegaconf import open_dict
from pyscf import gto

from mldft.ml.train import train
from mldft.ofdft.basis_integrals import get_normalization_vector
from mldft.ofdft.energies import Energies
from mldft.ofdft.optimizer import TorchOptimizer
from mldft.ofdft.run_density_optimization import SampleGenerator, run_singlepoint_ofdft


@pytest.fixture(scope="module")
def generate_run_path(tmp_path_factory):
    """Generate run path and model files for test."""
    torch.set_default_dtype(torch.float32)
    run_path = tmp_path_factory.mktemp("run_path") / "model"
    run_path.mkdir()

    with initialize(version_base="1.3", config_path="../../configs/ml"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "data=two_electron",
                "data/transforms=no_basis_transforms",
                # "data/transforms=global_natrep",
                # "data/transforms=local_frames_global_natrep",
                "model=graphformer_small",
            ],
        )
        # set defaults for all tests
        with open_dict(cfg):
            cfg.data.datamodule.batch_size = 2
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 2
            cfg.trainer.limit_val_batches = 1
            cfg.trainer.limit_test_batches = 1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.paths.output_dir = str(run_path)
            cfg.paths.log_dir = str(run_path)
            cfg.paths.work_dir = str(run_path)
    train(cfg)

    return run_path


def test_run_single_point_ofdft(generate_run_path):
    """Test the run_singlepoint_ofdft function with a simple molecule."""
    run_path = generate_run_path

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", unit="angstrom")

    model_config = SampleGenerator.from_run_path(run_path, device="cpu")

    func_factory = model_config.get_functional_factory()

    optimizer = TorchOptimizer(torch.optim.SGD, 1e-4, 10, lr=3e-3, momentum=0.9)
    energies, coeffs, converged = run_singlepoint_ofdft(
        mol, model_config, func_factory, optimizer=optimizer
    )

    sample = model_config.get_sample_from_mol(mol)
    normalization_vector = torch.as_tensor(get_normalization_vector(sample.mol))

    assert not converged
    assert torch.isclose(
        normalization_vector @ coeffs,
        torch.as_tensor(mol.nelectron, dtype=torch.float64),
    )
    assert isinstance(energies, Energies)
