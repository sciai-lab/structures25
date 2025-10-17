import h5py
import pytest
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from mldft.datagen.kohn_sham_dataset import compute_kohn_sham_dataset


@pytest.fixture
def cfg_datagen(tmp_path) -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for data generation.

    Returns:
        A DictConfig object containing a default Hydra configuration for data generation.
    """
    with initialize(version_base="1.3", config_path="../../configs/datagen"):
        cfg = compose(
            config_name="config.yaml",
            return_hydra_config=True,
            overrides=[
                "hydra.job.num=0",
                f"hydra.runtime.output_dir={tmp_path}",
                "hydra/job_logging=disabled",
                "hydra/hydra_logging=disabled",
            ],
        )
    return cfg


def get_dataset_config(yaml_name):
    with initialize(version_base="1.3", config_path="../../configs/datagen"):
        dataset_config = compose(
            config_name="config.yaml",
            return_hydra_config=True,
            overrides=[
                f"dataset={yaml_name.split('.')[0]}",
            ],
        )
        dataset_config = dataset_config.dataset
    return dataset_config


def modify_datagen_config(cfg_datagen, dataset_config, dataset):
    cfg_datagen.dataset = dataset_config
    with open_dict(cfg_datagen):
        # breakpoint()
        cfg_datagen.dataset.raw_data_dir = dataset.raw_data_dir
        cfg_datagen.dataset.kohn_sham_data_dir = dataset.kohn_sham_data_dir
        cfg_datagen.dataset.filename = dataset.filename
        cfg_datagen.dataset.name = dataset.name
        cfg_datagen.num_processes = 1
        cfg_datagen.n_molecules = 2

        if "subset" in dataset_config.keys():
            cfg_datagen.dataset["subset"] = dataset_config["subset"]
    return cfg_datagen


def assert_datagen_files(cfg_datagen, dataset, already_present_files):
    expected_yaml_files = 2
    no_files_expected = 2 + cfg_datagen.n_molecules
    written_files = list(dataset.kohn_sham_data_dir.glob("*"))
    assert (
        len(written_files) - len(already_present_files) == no_files_expected
    ), f"More or less than {no_files_expected} files written."
    yaml_files = sorted(list(dataset.kohn_sham_data_dir.glob("*.yaml")))
    chk_files = list(dataset.kohn_sham_data_dir.glob("*.chk"))
    assert (
        len(yaml_files) == expected_yaml_files
    ), "0 or more than one yaml file found in kohn_sham_data_dir."
    assert (
        len(chk_files) - len(already_present_files) == no_files_expected - expected_yaml_files
    ), f"0 or more than {no_files_expected - 2} chk file found in kohn_sham_data_dir."
    assert yaml_files[0].name == "atomic_numbers.yaml", "yaml file does not have the correct name."
    assert yaml_files[1].name == "config.yaml", "yaml file does not have the correct name."
    atomic_numbers = OmegaConf.load(yaml_files[0])
    assert isinstance(atomic_numbers, ListConfig), "Atomic numbers should be a ListConfig."
    config = OmegaConf.load(yaml_files[1])
    assert "basis" in config.keys(), "No basis key in config.yaml."
    assert "xc" in config.keys(), "No xc key in config.yaml."
    assert "initialization" in config.keys(), "No initialization key in config.yaml."
    assert chk_files[0].name.startswith(dataset.filename), "chk file does not start with filename."
    with h5py.File(chk_files[0], "r") as f:
        assert "Results" in f, "No Results group in chk file."
        assert f"KS-iteration/{0:d}" in f, "No KS-iteration group in chk file."
    # Clean up written files
    for f in written_files:
        f.unlink()


dataset_settings = [
    ("qm9_dataset", "qm9.yaml"),
]


@pytest.mark.slow
@pytest.mark.parametrize("dataset,yaml_name", dataset_settings)
def test_kohn_sham_single_process(cfg_datagen, dataset, yaml_name, request):
    """Test the kohn_sham_parallel function with a single process."""
    dataset = request.getfixturevalue(dataset)
    HydraConfig().set_config(cfg_datagen)

    dataset_config = get_dataset_config(yaml_name)

    cfg_datagen = modify_datagen_config(cfg_datagen, dataset_config, dataset)

    with open_dict(cfg_datagen):
        cfg_datagen.num_processes = 1

    already_present_files = list(dataset.kohn_sham_data_dir.glob("*"))
    logger.disable("mldft")
    # Try to compute one molecule with a single process
    try:
        compute_kohn_sham_dataset(cfg_datagen)
    except Exception as e:
        pytest.fail(f"Function raised an exception: {e}")

    assert_datagen_files(cfg_datagen, dataset, already_present_files)


@pytest.mark.slow
@pytest.mark.parametrize("dataset,yaml_name", dataset_settings)
def test_kohn_sham_two_processes(cfg_datagen, dataset, yaml_name, request):
    """Test the kohn_sham_parallel function with two processes."""
    dataset = request.getfixturevalue(dataset)
    HydraConfig().set_config(cfg_datagen)

    dataset_config = get_dataset_config(yaml_name)

    cfg_datagen = modify_datagen_config(cfg_datagen, dataset_config, dataset)

    with open_dict(cfg_datagen):
        cfg_datagen.num_processes = 2

    already_present_files = list(dataset.kohn_sham_data_dir.glob("*"))
    logger.disable("mldft")
    # Try to compute one molecule with a single process
    try:
        compute_kohn_sham_dataset(cfg_datagen)
    except Exception as e:
        pytest.fail(f"Function raised an exception: {e}")

    assert_datagen_files(cfg_datagen, dataset, already_present_files)
