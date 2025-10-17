import os
import pickle
import tempfile
import time
from functools import partial
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.multiprocessing as mp
from dotenv import load_dotenv
from hydra.utils import instantiate
from loguru import logger
from matplotlib import pyplot as plt
from omegaconf import DictConfig, ListConfig, OmegaConf
from pyscf import gto
from tqdm import tqdm

import mldft.utils.omegaconf_resolvers  # noqa
from mldft.ml.data.components.basis_transforms import (
    ApplyBasisTransformation,
    transform_tensor,
)
from mldft.ml.data.components.convert_transforms import (
    PrepareForDensityOptimization,
    str_to_torch_float_dtype,
    to_torch,
)
from mldft.ml.data.components.dataset import OFDataset
from mldft.ml.data.components.loader import OFLoader
from mldft.ml.data.components.of_data import BasisInfo, OFData, Representation
from mldft.ml.models.mldft_module import MLDFTLitModule
from mldft.ml.preprocess.dataset_statistics import DatasetStatistics
from mldft.ofdft.callbacks import ConvergenceCallback
from mldft.ofdft.density_optimization import (
    density_optimization,
    density_optimization_with_label,
)
from mldft.ofdft.energies import Energies
from mldft.ofdft.functional_factory import FunctionalFactory, requires_grid
from mldft.ofdft.optimizer import DEFAULT_DENSITY_OPTIMIZER, Optimizer
from mldft.utils import extras
from mldft.utils.environ import get_mldft_data_path, get_mldft_model_path
from mldft.utils.instantiators import instantiate_model
from mldft.utils.molecules import check_atom_types
from mldft.utils.pdf_utils import HierarchicalPlotPDF
from mldft.utils.plotting.density_optimization import plot_density_optimization
from mldft.utils.plotting.summary_density_optimization import (
    density_optimization_summary_pdf_plot,
    get_runwise_density_optimization_data,
    save_density_optimization_metrics,
)
from mldft.utils.pyscf_pretty_print import mole_to_sum_formula
from mldft.utils.sad_guesser import SADNormalizationMode
from mldft.utils.utils import set_default_torch_dtype


def parse_run_path(run_path: Path | str) -> Path:
    """Parse the run path, making it absolute if necessary."""
    run_path = Path(str(run_path))
    if not run_path.is_absolute():
        run_path = get_mldft_model_path() / "train" / "runs" / run_path
    return run_path


def run_to_checkpoint_path(run_path: Path | str, use_last_ckpt: bool = True) -> Path:
    """Get the path to the checkpoint of a run."""
    if use_last_ckpt:
        checkpoint_path = run_path / "checkpoints" / "last.ckpt"
    else:
        checkpoint_dir = run_path / "checkpoints"
        checkpoint_list = list(checkpoint_dir.glob("epoch*.ckpt"))
        assert len(checkpoint_list) == 1, (
            f"Found {len(checkpoint_list)} checkpoints starting with epoch in {checkpoint_dir},"
            f"but need one to infer the best checkpoint."
        )
        checkpoint_path = checkpoint_list[0]
    return checkpoint_path


def add_density_optimization_trajectories_to_sample(
    sample: OFData,
    callback: ConvergenceCallback,
    energies_label: Energies,
    basis_info: BasisInfo,
    save_coeff_interval: int = 100,
):
    """Add the density optimization trajectories of energies and coefficients to the sample."""

    sample.add_item("stopping_index", callback.stopping_index, representation=Representation.NONE)
    sample.add_item(
        "trajectory_gradient_norm",
        torch.as_tensor(callback.gradient_norm, dtype=torch.float64),
        representation=Representation.SCALAR,
    )
    sample.add_item(
        "trajectory_l2_norm",
        torch.as_tensor(callback.l2_norm, dtype=torch.float64),
        representation=Representation.SCALAR,
    )
    sample.add_item(
        "trajectory_energy_electronic",
        torch.as_tensor([e.electronic_energy for e in callback.energy], dtype=torch.float64),
        representation=Representation.SCALAR,
    )
    sample.add_item(
        "trajectory_energy_total",
        torch.as_tensor([e.total_energy for e in callback.energy], dtype=torch.float64),
        representation=Representation.SCALAR,
    )
    # callback.energy is a list of Energies objects. Assume every iteration has the same keys.
    for energy_name in callback.energy[0].energies_dict.keys():
        sample.add_item(
            "trajectory_energy_" + energy_name,
            torch.as_tensor([e[energy_name] for e in callback.energy], dtype=torch.float64),
            representation=Representation.SCALAR,
        )
    # This is just for convenience as of yet
    sample.add_item(
        "ground_state_energy_total",
        energies_label.total_energy,  # float64?
        representation=Representation.SCALAR,
    )
    sample.add_item(
        "ground_state_energy_electronic",
        energies_label.electronic_energy,
        representation=Representation.SCALAR,
    )
    for energy_name, ground_state_energy in energies_label.energies_dict.items():
        sample.add_item(
            "ground_state_energy_" + energy_name,
            ground_state_energy,
            representation=Representation.SCALAR,
        )

    coeff_indices = torch.arange(0, len(callback.coeffs), save_coeff_interval)
    sample.add_item("save_coeff_interval", save_coeff_interval, representation=Representation.NONE)
    sample.add_item(
        "trajectory_coeffs",
        torch.stack([callback.coeffs[i] for i in coeff_indices]),
        representation=Representation.VECTOR,
    )
    sample.add_item(
        "predicted_ground_state_coeffs",
        callback.coeffs[callback.stopping_index].detach().clone(),
        representation=Representation.VECTOR,
    )
    return sample


def configure_dataset_indices(
    dataset_size: int,
    n_molecules: int,
    molecule_choice: str | list,
    seed: int | None,
    start_idx: int = 0,
) -> np.ndarray:
    """Configure the indices of the dataset to optimize."""
    if n_molecules > dataset_size:
        logger.warning(
            f"Requested {n_molecules} molecules, but only {dataset_size} are available. "
            f"Setting n_molecules to {dataset_size}."
        )
        n_molecules = dataset_size
    if molecule_choice == "first_n":
        dataset_indices = np.arange(n_molecules)[start_idx:]
    elif molecule_choice == "random":
        if start_idx != 0:
            logger.warning(
                f"Random molecule choice with start_idx {start_idx}. Ignoring start_idx."
            )
        dataset_indices = np.random.choice(dataset_size, n_molecules, replace=False)
    elif molecule_choice == "seeded_random":
        np.random.seed(seed)
        # When calling this with different n_molecules, the order stays the same
        dataset_indices = np.random.choice(dataset_size, n_molecules, replace=False)[start_idx:]
    elif isinstance(molecule_choice, (list, ListConfig)):
        if len(molecule_choice) < n_molecules:
            logger.warning(
                f"Requested {n_molecules} molecules, but only {len(molecule_choice)} are available. "
                f"Setting n_molecules to {len(molecule_choice)}."
            )
            n_molecules = len(molecule_choice)
        elif len(molecule_choice) > n_molecules:
            logger.warning(
                f"Requested {n_molecules} molecules, but {len(molecule_choice)} are available. "
                f"Using the first {n_molecules} molecules."
            )
        dataset_indices = np.array(molecule_choice[start_idx:n_molecules])
    elif isinstance(molecule_choice, str) and molecule_choice.startswith("always_"):
        index = int(molecule_choice.split("_")[1])
        dataset_indices = np.array([index] * n_molecules)
    else:
        raise ValueError(
            f"Unknown molecule choice {molecule_choice}. "
            f"Must be 'first_n', 'random', 'seeded_random', or a list of indices."
        )
    return dataset_indices


def set_torch_defaults_worker(id: int, num_threads: int, device: torch.device | str):
    """Set the torch defaults for a dataloader worker."""
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(device)
    torch.set_num_threads(num_threads)


def worker(
    process_idx: int,
    dataset: OFDataset,
    basis_info: BasisInfo,
    checkpoint_path: Path,
    guess_path: Path | None,
    optimizer: Optimizer,
    device: str | torch.device,
    transform_device: str | torch.device,
    num_workers: int,
    num_threads: int,
    model_dtype: str | torch.dtype,
    xc_functional: str,
    negative_integrated_density_penalty_weight: float,
    use_last_ckpt: bool,
    initialization: str,
    dataset_statistics_path: Path | str,
    convergence_criterion: str,
    plot_queue,
    plot_every_n: int,
    save_dir: Path,
    save_denop_samples: bool,
    fail_fast: bool,
    save_coeff_interval: int,
):
    """Worker process for density optimization."""
    os.environ["DENOP_PID"] = str(process_idx)
    torch.set_default_dtype(torch.float64)
    if (num_workers == 0) and (transform_device != device):
        logger.warning(
            f"Setting default device to the transform device ({transform_device}) since num_workers=0."
        )
        torch.set_default_device(transform_device)
    torch.set_num_threads(num_threads)
    logger.remove()
    logger_format = (
        "<green>{time:HH:mm:ss}</green>|<level>{level: <8}</level>|<level>{message}</level>"
    )
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        format=logger_format,
        colorize=True,
        enqueue=True,
    )
    dataset_length = len(dataset)
    dataloader = OFLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=1 if num_workers > 0 else None,
        list_keys=[],
        worker_init_fn=partial(
            set_torch_defaults_worker, num_threads=num_threads, device=transform_device
        ),
    )
    lightning_module = MLDFTLitModule.load_from_checkpoint(checkpoint_path, map_location=device)
    lightning_module.eval()
    lightning_module.to(model_dtype)

    if guess_path is not None:
        guess_path = parse_run_path(guess_path)
        guess_checkpoint_path = run_to_checkpoint_path(guess_path, use_last_ckpt)
        proj_minao_module = MLDFTLitModule.load_from_checkpoint(
            guess_checkpoint_path, map_location=device
        )
        proj_minao_module.eval()
        proj_minao_module.to(torch.float64)
    else:
        proj_minao_module = lightning_module

    if initialization in ["sad", "sad_transformed"]:
        sad_guess_kwargs = dict(
            dataset_statistics=DatasetStatistics(dataset_statistics_path),
            normalization_mode=SADNormalizationMode.PER_ATOM_WEIGHTED,
            basis_info=basis_info,
            weigher_key="ground_state_only",
            spherical_average=True,
        )
    else:
        sad_guess_kwargs = {}

    func_factory = FunctionalFactory.from_module(
        lightning_module,
        xc_functional,
        negative_integrated_density_penalty_weight=negative_integrated_density_penalty_weight,
    )
    start_time = time.time()
    for i, sample in enumerate(dataloader):
        try:
            sample = sample.to_data_list()[0]
            sample.to(device)
            logger.info(
                f"{'Optimizing':<11}{mole_to_sum_formula(sample.mol, True):<16}"
                f"{sample.mol_id:<15} on worker {process_idx} ({i + 1}/{dataset_length}), "
            )
            callback = ConvergenceCallback()
            (
                metric_dict,
                callback,
                energies_label,
                energy_functional,
            ) = density_optimization_with_label(
                sample,
                sample.mol,
                optimizer,
                func_factory,
                callback,
                proj_minao_module=proj_minao_module,
                initial_guess_str=initialization,
                sad_guess_kwargs=sad_guess_kwargs,
                convergence_criterion=convergence_criterion,
            )
            end_time = time.time()
            time_per_mol = end_time - start_time
            sample.add_item("time", time_per_mol, representation=Representation.SCALAR)
            if save_denop_samples:
                # Delete large matrices
                sample.delete_item("overlap_matrix")
                sample.delete_item("coulomb_matrix")
                sample.delete_item("nuclear_attraction_vector")
                if hasattr(sample, "ao"):
                    sample.delete_item("ao")
                # Transform back to untransformed basis
                coeffs_callback = torch.stack(callback.coeffs)  # (n_iterations, n_coeffs)
                coeffs_callback_untransformed = transform_tensor(
                    coeffs_callback.t(),
                    transformation_matrix=sample.inv_transformation_matrix.cpu(),
                    inv_transformation_matrix=sample.transformation_matrix.cpu(),
                    representation=Representation.VECTOR,
                ).t()
                # This will turn the transformation matrix to the identity matrix
                sample = ApplyBasisTransformation()(
                    sample,
                    sample.transformation_matrix,
                    sample.inv_transformation_matrix,
                    invert=True,
                )
                # Go to cpu and list of tensors for compatibility with callback class
                callback.coeffs = coeffs_callback_untransformed.cpu().unbind(dim=0)
                sample.delete_item("transformation_matrix")
                sample.delete_item("inv_transformation_matrix")
                sample = sample.to("cpu")
                sample = add_density_optimization_trajectories_to_sample(
                    sample, callback, energies_label, basis_info, save_coeff_interval
                )
                torch.save(
                    sample,
                    save_dir / "sample_trajectories" / f"sample_{sample.mol_id}.pt",
                )
            if plot_queue is not None and i % plot_every_n == 0:
                plot_path = tempfile.gettempdir() + f"/sample_{sample.mol_id}.pt"
                torch.save((sample, callback, energies_label), plot_path)
                plot_queue.put(plot_path)
            start_time = time.time()
        except KeyboardInterrupt:
            logger.warning("Received KeyboardInterrupt. Exiting.")
        except Exception as e:
            logger.exception(f"Error in worker {process_idx} during density optimization: {e}")
            if fail_fast:
                raise e
            else:
                logger.warning(
                    f"Skipping molecule {sample.mol_id} due to error. Set fail_fast=True to raise immediately."
                )
                continue
    return


def plotting_worker(
    plot_queue: mp.Queue,
    save_dir: Path,
    basis_info: BasisInfo,
    enable_grid_plots: bool,
    save_individual_plots: bool = False,
    num_threads: int = 8,
    fail_fast: bool = False,
):
    """Worker process for handling plotting."""
    logger.remove()
    logger_format = (
        "<green>{time:HH:mm:ss}</green>|<level>{level: <8}</level>|<level>{message}</level>"
    )
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        format=logger_format,
        colorize=True,
        enqueue=True,
    )
    torch.set_num_threads(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    with HierarchicalPlotPDF(
        out_pdf_path=save_dir / "density_optimization.pdf",
        individual_plot_directory=(
            save_dir / "individual_results" if save_individual_plots else None
        ),
    ) as pdf:
        while True:
            plot_path = None
            try:
                plot_path = plot_queue.get()
                if plot_path is None:  # sentinel value
                    break

                # Load plotting data
                sample, callback, energies_label = torch.load(
                    plot_path, map_location="cpu", weights_only=False
                )
                coeff_dim = sample.ground_state_coeffs.shape[0]
                sample.add_item(
                    "transformation_matrix",
                    torch.eye(coeff_dim, dtype=torch.float64),
                    representation=Representation.VECTOR,
                )
                sample.add_item(
                    "inv_transformation_matrix",
                    torch.eye(coeff_dim, dtype=torch.float64),
                    representation=Representation.VECTOR,
                )
                # Create plots
                basis_l = basis_info.l_per_basis_func[sample.basis_function_ind]
                plot_density_optimization(
                    callback,
                    energies_label,
                    sample.ground_state_coeffs,
                    sample,
                    callback.stopping_index,
                    basis_l,
                    enable_grid_operations=enable_grid_plots,
                )
                plt.suptitle(
                    f"Density Optimization for {sample.mol_id}, {mole_to_sum_formula(sample.mol, True)}"
                )
                plt.tight_layout(rect=(0, 0.03, 1, 0.97))
                pdf.savefig(f"molecule: {mole_to_sum_formula(sample.mol, True)}")
                plt.close()
                # Clean up temporary file
                os.unlink(plot_path)

            except KeyboardInterrupt:
                if plot_path is not None:
                    try:
                        os.unlink(plot_path)
                    except OSError:
                        pass
                break
            except Exception as e:
                logger.exception(f"Error in plotting worker: {e}")
                if plot_path is not None:
                    try:
                        os.unlink(plot_path)
                    except OSError:
                        pass
                if fail_fast:
                    raise e


def evaluate_density_optimization(
    save_dir: Path,
    n_molecules: int | None = None,
    energy_names: str = None,
    plot_l1_norm: bool = True,
    l1_grid_level: int = 3,
    l1_grid_prune: str = "nwchem_prune",
    swarm_plot_subsample: float = 1.0,
):
    run_data_dict = get_runwise_density_optimization_data(
        sample_dir=save_dir / "sample_trajectories",
        n_molecules=n_molecules,
        energy_names=energy_names,
        plot_l1_norm=plot_l1_norm,
        l1_grid_level=l1_grid_level,
        l1_grid_prune=l1_grid_prune,
    )
    save_density_optimization_metrics(
        output_path=save_dir / "density_optimization_metrics.yaml",
        run_data_dict=run_data_dict,
    )
    density_optimization_summary_pdf_plot(
        out_pdf_path=save_dir / "density_optimization_summary.pdf",
        run_data_dict=run_data_dict,
        subsample=swarm_plot_subsample,
    )


@set_default_torch_dtype(torch.float64)
def run_ofdft(
    run_path: Path | str,
    optimizer: Optimizer,
    guess_path: Path | str | None = None,
    use_last_ckpt: bool = True,
    device: torch.device | str = "cpu",
    transform_device: torch.device | str = "cpu",
    num_processes_per_device: int = 1,
    num_devices: int = 1,
    num_workers: int = 1,
    num_threads_per_process: int = 8,
    model_dtype: torch.dtype = torch.float64,
    xc_functional: str = "PBE",
    initialization: str = "minao",
    n_molecules: int = 1,
    molecule_choice: str | list[int] = "first_n",
    seed: int = 0,
    log_file: str | None = None,
    save_individual_plots: bool = False,
    save_denop_samples: bool = False,
    plot_every_n: int = 10,
    swarm_plot_subsample: float = 1.0,
    ofdft_kwargs: dict = None,
    split: str = "val",
    split_file_path: str | None = None,
    plot_l1_norm: bool = True,
    enable_grid_operations: bool = True,
    l1_grid_level: int = 3,
    l1_grid_prune: str = "nwchem_prune",
    negative_integrated_density_penalty_weight: float = 0.0,
    convergence_criterion: str = "last_iter",
    fail_fast: bool = False,
    save_coeff_interval: int = 100,
):
    """Script to run ofdft using a model checkpoint on multiple molecules.

    Note: Right now this only supports density optimizations using a checkpoint.

    Args:
        run_path (Path | str): The path to the run directory.
        guess_path (Path | str, optional): The path to the guess directory. Defaults to None, then the same model
            is used for the proj_minao guess.
        use_last_ckpt (bool, optional): Whether to use the last checkpoint. Defaults to True.
        device (torch.device | str, optional): The device to run on. Defaults to "cpu".
        model_dtype (torch.dtype, optional): The dtype of the model. Defaults to torch.float64.
        xc_functional (str, optional): The XC functional to use. Defaults to "PBE". Irrelevant if the xc functional
            is part of the model prediction.
        initialization (str, optional): The initialization to use. Defaults to "minao". Other possible values include
            "hueckel", "proj_minao", "label".
        optimizer (str, optional): The optimizer to use, e.g. "gradient_descent" or "slsqp".
            Defaults to "gradient_descent".
        n_molecules (int, optional): The number of molecules to optimize. Defaults to 1.
        molecule_choice (str | list[int], optional): The choice of molecules to optimize. Options are
            "first_n", "random", "seeded_random", or a list of indices. Defaults to "first_n".
        log_file (str | None, optional): The path to the log file. Defaults to None, then no log file is created.
        save_individual_plots (bool, optional): Whether to keep individual plots for each molecule. Defaults to False.
        save_denop_samples (bool, optional): Whether to save the density optimization trajectories of the samples.
            Defaults to False.
        swarm_plot_subsample (float, optional): The subsample factor for the swarm plots. Defaults to 1.0.
        ofdft_kwargs (dict, optional): Additional keyword arguments for the OFDFT class. Defaults to None.
        split (str, optional): The split to use, i.e. "val" or "train". Defaults to "val".
        plot_l1_norm (bool, optional): Whether to plot the L1 norm of the density error, for which the integration
            grid is required. Defaults to True.
        l1_grid_level (int, optional): The grid level of the integration grid for the L1 norm. Defaults to 0.
        l1_grid_prune (str, optional): The pruning method for the integration grid for the L1 norm.
            Defaults to "nwchem_prune".
        negative_integrated_density_penalty_weight (float, optional): The weight of the negative integrated density
            penalty. Defaults to 0.0.
        convergence_criterion (str, optional): The convergence criterion for the density optimization.
        fail_fast (bool, optional): Whether to raise an error immediately if a molecule fails. Defaults to False,
            such that errors are logged, but the script continues. Useful to set to True for debugging.
    """
    mp.set_start_method("spawn", force=True)
    ofdft_kwargs = dict() if ofdft_kwargs is None else ofdft_kwargs
    if negative_integrated_density_penalty_weight != 0.0 and not enable_grid_operations:
        raise ValueError(
            "Cannot use negative integrated density penalty without grid operations. "
            "Set enable_grid_operations to True."
        )

    logger.remove()
    logger_format = (
        "<green>{time:HH:mm:ss}</green>|<level>{level: <8}</level>|<level>{message}</level>"
    )
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        format=logger_format,
        colorize=True,
        enqueue=True,
    )
    if log_file is not None:
        log_file = Path(log_file)
        logger.add(
            log_file,
            level="TRACE",
            rotation="10 MB",
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )
        logger.info(f'Logging to "{log_file}"')
    run_path = parse_run_path(run_path)
    model_config_path = run_path / "hparams.yaml"
    model_config = OmegaConf.load(model_config_path)
    transforms = instantiate(model_config.data.transforms)
    basis_info = instantiate(model_config.data.basis_info)
    add_grid = requires_grid(
        model_config.data.target_key, negative_integrated_density_penalty_weight
    )
    transforms.pre_transforms.insert(
        0, PrepareForDensityOptimization(basis_info, add_grid=add_grid)
    )
    transforms.add_transformation_matrix = True
    transforms.use_cached_data = False
    if split_file_path is None:
        split_file_path = Path(model_config.data.datamodule.split_file)
    dataset_kwargs = instantiate(model_config.data.datamodule.dataset_kwargs)
    dataset_statistics_path = model_config.data.dataset_statistics.path
    if initialization == "sad":
        dataset_statistics_path = dataset_statistics_path.replace(
            model_config.data.transforms.name, "no_basis_transforms"
        )

    checkpoint_path = run_to_checkpoint_path(run_path, use_last_ckpt)
    num_processes = num_processes_per_device * num_devices
    if device == "cpu":
        assert num_devices == 1, "Only one cpu device is supported."
    elif device == "cuda":
        assert (
            num_devices <= torch.cuda.device_count()
        ), f"Configured {num_devices} cuda devices but only {torch.cuda.device_count()} are available."
    save_dir = log_file.parent
    if save_denop_samples:
        save_dir.mkdir(exist_ok=True)  # for saving the denop trajectories of the samples
        (save_dir / "sample_trajectories").mkdir(exist_ok=True)
        torch.save(basis_info, save_dir / "sample_trajectories" / "basis_info.pt")

    data_dir = get_mldft_data_path()
    with open(split_file_path, "rb") as f:
        split_dict = pickle.load(f)
    label_subdir = "labels"
    val_paths = [
        data_dir / dataset / label_subdir / label_path
        for dataset, label_path, _ in split_dict[split]
    ]
    val_iterations = [scf_iterations for _, _, scf_iterations in split_dict[split]]
    dataset_kwargs.update(
        {
            "limit_scf_iterations": -1,
            "additional_keys_at_ground_state": {
                "of_labels/energies/e_electron": Representation.SCALAR,
                "of_labels/energies/e_ext": Representation.SCALAR,
                "of_labels/energies/e_hartree": Representation.SCALAR,
                "of_labels/energies/e_kin": Representation.SCALAR,
                "of_labels/energies/e_kin_plus_xc": Representation.SCALAR,
                "of_labels/energies/e_kin_minus_apbe": Representation.SCALAR,
                "of_labels/energies/e_kinapbe": Representation.SCALAR,
                "of_labels/energies/e_xc": Representation.SCALAR,
                "of_labels/energies/e_tot": Representation.SCALAR,
            },
        }
    )
    if plot_every_n > 0:
        plot_queue = mp.Queue()
        plot_process = mp.Process(
            target=plotting_worker,
            args=(
                plot_queue,
                save_dir,
                basis_info,
                enable_grid_operations,
                save_individual_plots,
                num_threads_per_process,
                fail_fast,
            ),
        )
        plot_process.start()
    else:
        plot_queue = None

    dataset_indices = configure_dataset_indices(len(val_paths), n_molecules, molecule_choice, seed)
    processes = []
    dataset_indices = np.array_split(dataset_indices, num_processes)
    for i in range(num_processes):
        if device == "cuda":
            process_device = f"cuda:{i % num_devices}"
        else:
            process_device = "cpu"
        if transform_device == "cuda":
            transforms.device = f"cuda:{i % num_devices}"
        dataset = OFDataset(
            paths=[val_paths[j] for j in dataset_indices[i]],
            num_scf_iterations_per_path=[val_iterations[j] for j in dataset_indices[i]],
            basis_info=basis_info,
            transforms=transforms,
            **dataset_kwargs,
        )
        p = mp.Process(
            target=worker,
            args=(
                i,
                dataset,
                basis_info,
                checkpoint_path,
                guess_path,
                optimizer,
                process_device,
                transform_device,
                num_workers,
                num_threads_per_process,
                model_dtype,
                xc_functional,
                negative_integrated_density_penalty_weight,
                use_last_ckpt,
                initialization,
                dataset_statistics_path,
                convergence_criterion,
                plot_queue,
                plot_every_n,
                save_dir,
                save_denop_samples,
                fail_fast,
                save_coeff_interval,
            ),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    # Signal plotting process to finish
    if plot_every_n > 0:
        plot_queue.put(None)
        plot_process.join()

    evaluate_density_optimization(
        save_dir,
        n_molecules,
        plot_l1_norm=plot_l1_norm,
        l1_grid_level=l1_grid_level,
        l1_grid_prune=l1_grid_prune,
        swarm_plot_subsample=swarm_plot_subsample,
    )


def calculate_basis_size(mol: gto.Mole, basis_info: BasisInfo) -> int:
    """Calculate the size of the basis set for a given molecule.

    Args:
        mol: The molecule.
        basis_info: The basis information object.

    Returns:
        The number of basis functions.
    """
    n_basis = 0
    for atom_number in mol.atom_charges():
        atom_index = basis_info.atomic_number_to_atom_index[atom_number]
        n_basis += basis_info.basis_dim_per_atom[atom_index]

    return n_basis


class SampleGenerator:
    """Class to generate samples from the model configuration.

    Attributes:
        model_config: The model configuration.
        model: The model.
        transforms: The transforms.
        basis_info: The basis information.
        negative_integrated_density_penalty_weight: The weight for the negative integrated density penalty.
    """

    def __init__(
        self,
        model_config: DictConfig,
        model: MLDFTLitModule,
        negative_integrated_density_penalty_weight: float = 0.0,
        transform_device: str | torch.device = "cpu",
    ) -> None:
        """Initialize the SampleGenerator.

        Args:
            model_config: The model configuration.
            model: The model.
            negative_integrated_density_penalty_weight: The weight for the negative integrated density penalty.
        """
        basis_info = instantiate(model_config.data.basis_info)

        transforms = instantiate(model_config.data.transforms)
        add_grid = requires_grid(
            model_config.data.target_key, negative_integrated_density_penalty_weight
        )
        transforms.pre_transforms.insert(
            0, PrepareForDensityOptimization(basis_info, add_grid=add_grid)
        )
        transforms.add_transformation_matrix = True
        transforms.use_cached_data = False

        self.negative_integrated_density_penalty_weight = (
            negative_integrated_density_penalty_weight
        )
        self.model = model
        self.transforms = transforms
        self.model_config = model_config
        self.basis_info = basis_info

        if transform_device == "cuda":
            self.transforms.device = "cuda"

    @classmethod
    def from_run_path(
        cls,
        run_path: str | Path,
        device: str | torch.device = "cuda",
        transform_device: str | torch.device = "cpu",
        negative_integrated_density_penalty_weight: float = 0.0,
        use_last_ckpt: bool = True,
    ) -> "SampleGenerator":
        """Create a SampleGenerator from a run path.

        Args:
            run_path: The run path.
            device: The device to load the model on.
            transform_device: The device to apply the transforms on.
            negative_integrated_density_penalty_weight: The weight for the negative integrated density penalty.
            ckpt_choice:

        Returns:
            The instantiated SampleGenerator
        """
        torch.set_default_dtype(torch.float64)
        run_path = parse_run_path(run_path)
        checkpoint_path = run_to_checkpoint_path(run_path, use_last_ckpt=use_last_ckpt)
        model_config_path = run_path / "hparams.yaml"
        model_config = OmegaConf.load(model_config_path)
        model = instantiate_model(checkpoint_path, device)

        return cls(
            model_config,
            model,
            negative_integrated_density_penalty_weight,
            transform_device=transform_device,
        )

    def get_sample_from_mol(self, mol: gto.Mole) -> OFData:
        """Get a sample from a molecule with the appropriate transforms applied.

        Args:
            mol: The molecule.

        Returns:
            The OFData sample.
        """
        # check that the molecule only contains allowed atom types by comparing to the
        # basis info of the model
        check_atom_types(mol, self.basis_info.atomic_numbers)

        n_basis = calculate_basis_size(mol, self.basis_info)

        sample = OFData.construct_new(
            basis_info=self.basis_info,
            pos=mol.atom_coords(unit="Bohr"),
            atomic_numbers=mol.atom_charges(),
            coeffs=np.zeros(n_basis),
            dual_basis_integrals="infer_from_basis",
            add_irreps=True,
        )
        sample = self.transforms.forward(sample)
        sample.mol.charge = mol.charge
        sample = to_torch(sample, device=self.model.device)
        return sample

    def get_functional_factory(self, xc_functional: str | None = None) -> FunctionalFactory:
        """Get a functional factory for the model and its config.

        Args:
            xc_functional: The XC functional to use.

        Returns:
            The functional factory.
        """
        return FunctionalFactory.from_module(
            self.model,
            xc_functional,
            negative_integrated_density_penalty_weight=self.negative_integrated_density_penalty_weight,
        )


def run_singlepoint_ofdft(
    mol: gto.Mole,
    sample_generator: SampleGenerator,
    func_factory: FunctionalFactory,
    optimizer: Optimizer = DEFAULT_DENSITY_OPTIMIZER,
    initial_guess_str: str = "minao",
    callback: ConvergenceCallback | None = None,
    ofdft_kwargs=None,
    return_sample: bool = False,
) -> tuple[Energies, torch.Tensor, bool] | tuple[Energies, torch.Tensor, bool, OFData]:
    """Run a single-point OFDFT calculation for the given molecule.

    Args:
        mol: The molecule.
        sample_generator: The sample generator.
        func_factory: The functional factory.
        optimizer: The optimizer.
        initial_guess_str: The initial guess.
        callback: The callback.
        ofdft_kwargs: Additional keyword arguments for density optimization.
        return_sample: Whether to return the sample as well.

    Returns:
        The final energies, coefficients and whether the calculation converged.
        If return_sample is True, also returns the OFData sample.
    """
    if ofdft_kwargs is None:
        ofdft_kwargs = dict()
    sample = sample_generator.get_sample_from_mol(mol)
    final_energies, final_coeffs, converged, _ = density_optimization(
        sample,
        sample.mol,
        optimizer,
        func_factory,
        callback,
        initialization=initial_guess_str,
        **ofdft_kwargs,
    )
    if return_sample:
        return final_energies, final_coeffs, converged, sample
    else:
        return final_energies, final_coeffs, converged


@hydra.main(version_base="1.3", config_path="../../configs/ofdft", config_name="ofdft.yaml")
def main(cfg: DictConfig):
    """Main function to use hydra main.

    Enables to also run the meth:`run_ofdft` in code.
    """
    load_dotenv()

    extras(cfg)

    optimizer = instantiate(cfg.get("optimizer"))
    run_ofdft(
        run_path=cfg.run_path,
        optimizer=optimizer,
        guess_path=cfg.get("guess_path"),
        use_last_ckpt=cfg.get("use_last_ckpt"),
        device=cfg.get("device"),
        transform_device=cfg.get("transform_device"),
        num_processes_per_device=cfg.get("num_processes_per_device"),
        num_devices=cfg.get("num_devices"),
        num_workers=cfg.get("num_workers_per_process"),
        num_threads_per_process=cfg.get("num_threads_per_process"),
        model_dtype=str_to_torch_float_dtype(cfg.get("model_dtype", torch.float64)),
        xc_functional=cfg.get("xc_functional"),
        initialization=cfg.get("initialization"),
        n_molecules=cfg.get("n_molecules"),
        molecule_choice=cfg.get("molecule_choice"),
        seed=cfg.get("seed"),
        log_file=cfg.get("log_file"),
        save_denop_samples=cfg.get("save_denop_samples"),
        plot_every_n=cfg.get("plot_every_n"),
        swarm_plot_subsample=cfg.get("swarm_plot_subsample"),
        ofdft_kwargs=cfg.get("ofdft_kwargs"),
        plot_l1_norm=cfg.get("plot_l1_norm"),
        l1_grid_level=cfg.get("l1_grid_level"),
        l1_grid_prune=cfg.get("l1_grid_prune"),
        negative_integrated_density_penalty_weight=cfg.get(
            "negative_integrated_density_penalty_weight"
        ),
        enable_grid_operations=cfg.get("enable_grid_operations"),
        split=cfg.get("split"),
        split_file_path=cfg.get("split_file_path"),
        convergence_criterion=cfg.get("convergence_criterion"),
        fail_fast=cfg.get("fail_fast"),
        save_coeff_interval=cfg.get("save_coeff_interval"),
    )


if __name__ == "__main__":
    main()
