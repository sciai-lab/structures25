"""This file prepares config fixtures for other tests."""
import copy
import pickle
from pathlib import Path

import numpy as np
import pytest
import rootutils
import zarr
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.basis_transforms import MasterTransformation
from mldft.ml.data.components.convert_transforms import ToTorch, to_torch
from mldft.ml.data.components.dataset import OFDataset
from mldft.ml.data.components.loader import OFLoader
from mldft.ml.data.components.of_data import OFData
from mldft.ml.models.components.sample_weighers import ConstantSampleWeigher
from mldft.ml.preprocess.dataset_statistics import DatasetStatistics


@pytest.fixture(scope="module")
def dummy_basis_info() -> BasisInfo:
    """Create a dummy BasisInfo object containing information about an OF basis including elements
    H, C and O."""

    atomic_numbers = np.array([1, 6, 8], dtype=np.uint8)
    basis_dict = {"H": "6-31++G**", "C": "6-31++G**", "O": "6-31++G**"}
    irreps_per_atom = ["3x0e+1x1o", "4x0e+3x1o+1x2e", "4x0e+3x1o+1x2e"]

    # integrals over non-S basis functions are zero
    integrals = [
        np.array([1.1, 1.2, 1.3, 0, 0, 0], dtype=np.float64),
        np.array([2.1, 2.2, 2.3, 2.4] + [0] * 14, dtype=np.float64),
        np.array([3.1, 3.2, 3.3, 3.4] + [0] * 14, dtype=np.float64),
    ]

    return BasisInfo(
        atomic_numbers=atomic_numbers,
        basis_dict=basis_dict,
        irreps_per_atom=irreps_per_atom,
        integrals=integrals,
    )


def create_dummy_sample(path, basis_info, n_scf=20, n_atom=30, mol_id="qm9_123", **dataset_kwargs):
    """Create a sample for testing purposes, consistent with a given basis_info."""

    dataset_kwargs.setdefault("compressor", None)

    dummy_energies = np.random.randn(n_scf)

    root = zarr.open(path, mode="w")
    geometry = root.create_group("geometry")
    geometry.create_dataset("mol_id", data=mol_id, **dataset_kwargs)
    geometry.create_dataset("atom_pos", data=np.random.randn(n_atom, 3), **dataset_kwargs)
    assert n_atom >= 3, f"Need at least 3 atoms for testing local frames, got {n_atom}"
    basis_atomic_numbers = np.array(basis_info.atomic_numbers)
    atomic_numbers = np.concatenate(
        [
            # guarantee that there are always at least three heavy atoms
            np.random.choice(basis_atomic_numbers[basis_atomic_numbers > 1], size=3),
            np.random.choice(basis_atomic_numbers, size=n_atom - 3),
        ]
    )
    atom_ind = basis_info.atomic_number_to_atom_index[atomic_numbers]
    geometry.create_dataset(
        "atomic_numbers",
        data=atomic_numbers,
        dtype=np.uint8,
        **dataset_kwargs,
    )

    of_labels = root.create_group("of_labels")
    of_labels.create_dataset("n_scf_steps", data=n_scf, **dataset_kwargs)

    energies = of_labels.create_group("energies")
    energies.create_dataset("e_kin", data=dummy_energies, **dataset_kwargs)
    energies.create_dataset("e_ex", data=dummy_energies, **dataset_kwargs)
    energies.create_dataset("e_hartree", data=dummy_energies, **dataset_kwargs)
    energies.create_dataset("e_nuc_nuc", data=dummy_energies[0], **dataset_kwargs)

    spatial = of_labels.create_group("spatial")
    n_basis = np.sum(basis_info.basis_dim_per_atom[atom_ind])
    spatial.create_dataset(
        "coeffs", data=np.random.randn(n_scf, n_basis), chunks=(1, n_basis), **dataset_kwargs
    )
    spatial.create_dataset(
        "basis_integrals",
        data=np.concatenate(basis_info.integrals[atom_ind])[None].repeat(n_scf, 0),
        chunks=(1, n_basis),
        **dataset_kwargs,
    )
    spatial.create_dataset(
        "dual_basis_integrals",
        data=np.concatenate(basis_info.integrals[atom_ind])[None].repeat(n_scf, 0),
        chunks=(1, n_basis),
        **dataset_kwargs,
    )
    spatial.create_dataset(
        "grad_kin",
        data=np.random.randn(n_scf, n_basis),
        chunks=(1, n_basis),
        **dataset_kwargs,
    )
    spatial.create_dataset(
        "grad_xc", chunks=(1, n_basis), data=np.random.randn(n_scf, n_basis), **dataset_kwargs
    )

    ks_labels = root.create_group("ks_labels")
    ks_labels.create_dataset("basis", data="dummy_basis", **dataset_kwargs)

    energies = ks_labels.create_group("energies")
    energies.create_dataset("e_kin", data=dummy_energies, **dataset_kwargs)
    energies.create_dataset("e_xc", data=dummy_energies, **dataset_kwargs)
    energies.create_dataset("e_hartree", data=dummy_energies, **dataset_kwargs)
    energies.create_dataset("e_nuc_nuc", data=dummy_energies[0], **dataset_kwargs)


# run with 20 and 1 scf steps, to make sure that the loader works for both multiple and a single scf step
@pytest.fixture(scope="module", params=[20, 1], ids=["20 scf steps", "1 scf step"])
def dummy_sample_path(tmpdir_factory, dummy_basis_info, request):
    """Create a dummy sample and provide the path for testing purposes."""
    tmp_path = tmpdir_factory.mktemp("dummy_data.zarr")
    scf_steps = request.param
    np.random.seed(0)
    create_dummy_sample(tmp_path, basis_info=dummy_basis_info, n_scf=scf_steps)
    return tmp_path


@pytest.fixture(scope="module")
def dummy_dataset_path(tmpdir_factory, dummy_basis_info):
    """Create a dummy dataset for testing purposes."""
    np.random.seed(0)
    n_samples = 10
    parent_directory = tmpdir_factory.mktemp("dummy_dataset")
    dataset_name = parent_directory.basename
    directory = parent_directory / "labels"
    for i in range(n_samples):
        path = directory / f"dummy_sample_{i}.zarr"
        mol_id = f"mol_{i}"
        create_dummy_sample(
            path, dummy_basis_info, n_atom=np.random.randint(10, 15), n_scf=1 + i, mol_id=mol_id
        )
    split_dict = {
        "train": [[dataset_name, f"dummy_sample_{i}.zarr", 1 + i] for i in range(8)],
        "val": [[dataset_name, "dummy_sample_8.zarr", 9]],
        "test": [[dataset_name, "dummy_sample_9.zarr", 10]],
    }
    # Save the split file into the temporary directory
    split_file = parent_directory / "train_val_test_split.pkl"
    with split_file.open("wb") as f:
        pickle.dump(split_dict, f)
    return directory


@pytest.fixture(scope="function")
def dummy_sample(dummy_sample_path, dummy_basis_info):
    """A dummy sample for testing purposes."""
    # We need the copy otherwise the transforms from one test affect the next ones
    return copy.deepcopy(OFData.from_file(dummy_sample_path, 0, dummy_basis_info, add_irreps=True))


@pytest.fixture(scope="function")
def dummy_sample_torch(dummy_sample):
    """A dummy sample with the to_torch transform applied for testing purposes."""
    return to_torch(dummy_sample)


@pytest.fixture(scope="module")
def master_transform_to_torch():
    """A master transform for testing purposes."""
    return MasterTransformation(name="no_basis_transforms", pre_transforms=[ToTorch()])


@pytest.fixture(scope="module")
def dummy_dataset_torch(dummy_dataset_path, dummy_basis_info, master_transform_to_torch):
    """Create a dummy dataset with the to_torch transform for testing purposes."""
    return OFDataset.from_directory(
        dummy_dataset_path,
        basis_info=dummy_basis_info,
        add_irreps=True,
        transforms=master_transform_to_torch,
    )


@pytest.fixture(scope="module")
def dummy_dataset_numpy(dummy_dataset_path, dummy_basis_info):
    """Create a dummy dataset without any transforms for testing purposes."""
    return OFDataset.from_directory(
        dummy_dataset_path, basis_info=dummy_basis_info, add_irreps=True, transforms=[]
    )


@pytest.fixture(scope="module")
def dummy_dataset_statistics(dummy_dataset_torch, dummy_basis_info, tmp_path_factory):
    """Create a dummy dataset statistics object for testing purposes."""
    tmp_path = tmp_path_factory.mktemp("dummy_statistics")
    return DatasetStatistics.from_dataset(
        path=tmp_path / "dummy_statistics.zarr",
        dataset=dummy_dataset_torch,
        basis_info=dummy_basis_info,
        with_bias=True,
        atom_ref_fit_sample_weighers=[ConstantSampleWeigher()],
        initial_guess_sample_weighers=[ConstantSampleWeigher()],
    )


@pytest.fixture(scope="module")
def dummy_loader(dummy_dataset_torch):
    """Create a loader with batch size 10 and a final smaller batch."""
    return OFLoader(dummy_dataset_torch, batch_size=10, shuffle=False, drop_last=False)


@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for training.

    :return: A DictConfig object containing a default Hydra configuration for training.
    """
    with initialize(version_base="1.3", config_path="../../configs/ml"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "data=two_electron",
                "data/transforms=no_basis_transforms",
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
    return cfg


@pytest.fixture(scope="package")
def cfg_eval_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for evaluation.

    :return: A DictConfig containing a default Hydra configuration for evaluation.
    """
    with initialize(version_base="1.3", config_path="../../configs/ml"):
        cfg = compose(config_name="eval.yaml", return_hydra_config=True, overrides=["ckpt_path=."])

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.datamodule.num_workers = 0
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope="function")
def cfg_train(cfg_train_global: DictConfig, tmp_path: Path) -> DictConfig:
    """A pytest fixture built on top of the `cfg_train_global()` fixture, which accepts a temporary
    logging path `tmp_path` for generating a temporary logging path.

    N    This is called by each test which uses the `cfg_train` arg. Each test generates its own temporary logging path.

        :param cfg_train_global: The input DictConfig object to be modified.
        :param tmp_path: The temporary logging path.

        :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    cfg = cfg_train_global.copy()

    with open_dict(cfg):
        # Since we don't use hydra these need to be set manually
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.work_dir = str(tmp_path)

    yield cfg
    # In case a hydra instance was set, remove it
    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_eval(cfg_eval_global: DictConfig, tmp_path: Path) -> DictConfig:
    """A pytest fixture built on top of the `cfg_eval_global()` fixture, which accepts a temporary
    logging path `tmp_path` for generating a temporary logging path.

    This is called by each test which uses the `cfg_eval` arg. Each test generates its own temporary logging path.

    :param cfg_train_global: The input DictConfig object to be modified.
    :param tmp_path: The temporary logging path.

    :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    cfg = cfg_eval_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()
