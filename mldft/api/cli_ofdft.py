import argparse
import os
import sys
from dataclasses import asdict, dataclass, is_dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import torch
from loguru import logger
from pyscf import gto
from tqdm import tqdm

from mldft.api.instantiate_from_args import (
    BaseConfig,
    ModelConfig,
    OptimizerConfig,
    get_optimizer_from_optimizer_args,
    get_sample_generator_from_model_args,
    get_xyzfiles_from_base_args,
)
from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.basis_transforms import ApplyBasisTransformation
from mldft.ml.data.components.of_data import OFData, Representation
from mldft.ml.preprocess.dataset_statistics import DatasetStatistics
from mldft.ofdft.energies import Energies
from mldft.ofdft.optimizer import Optimizer
from mldft.ofdft.run_density_optimization import SampleGenerator, run_singlepoint_ofdft
from mldft.utils.sad_guesser import SADNormalizationMode

CLI_HEADER = f"""MLDFT
Version: {version("mldft")}

Contributors:
Roman Remme, Tobias Kaczun, Tim Ebert, Christof A. Gehrig, Dominik Geng, Gerrit Gerhartz,
Marc K. Ickler, Manuel V. Klockow, Peter Lippmann, Johannes S. Schmidt, Simon Wagner,
Andreas Dreuw, Fred A. Hamprecht

Please cite our work if you use MLDFT in your research
https://doi.org/10.1021/jacs.5c06219
"""

DEFAULT_DATASET_STATISTICS_PATH = (
    Path(os.environ["DFT_STATISTICS"])
    / "sciai-test-mol/dataset_statistics/dataset_statistics_labels_no_basis_transforms_e_kin_plus_xc.zarr/"
)

OFDFT_KWARGS = [
    "normalize_initial_guess",
    "ks_basis",
    "proj_minao_module",
    "sad_guess_kwargs",
    "disable_printing",
]

# Define a way to mark messages that should go only to file
FILE_ONLY_TAG = "file_only"


@dataclass(slots=True)
class OFDFTRunResult:
    """Container summarizing the outcome of a single OFDFT calculation."""

    xyz_path: Path
    energies: Energies
    converged: bool
    coeffs: torch.Tensor | None
    sample: OFData
    sample_path: Path | None


def console_filter(record: dict) -> bool:
    """Filter to exclude messages marked as file-only from console output.

    Args:
        record (dict): Log record.
    Returns:
        bool: True if the message should be logged to console, False otherwise.
    """
    # Exclude messages explicitly marked as file-only
    return FILE_ONLY_TAG not in record["extra"]


def extract_group(prefix: str, args: argparse.Namespace, cls: type | None = None):
    """Extract args starting with prefix_ and optionally instantiate ``cls``."""

    group_dict = {
        key.removeprefix(prefix + "_"): val
        for key, val in vars(args).items()
        if key.startswith(prefix + "_")
    }
    return cls(**group_dict) if cls is not None else group_dict


def _config_to_log_dict(config: Any) -> Mapping[str, Any] | None:
    """Convert config-like objects into a mapping suitable for logging."""

    if config is None:
        return None
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, Mapping):
        return config
    if hasattr(config, "__dict__"):
        return vars(config)
    return {"value": config}


def log_config(base_args, model_args, optimizer_args, logger):
    """Log the configuration to the logger."""

    logger.info("CONFIGURATION:")
    for name, cfg in zip(
        ["Base args", "Model args", "Optimizer args"],
        [base_args, model_args, optimizer_args],
    ):
        cfg_dict = _config_to_log_dict(cfg)
        logger.info(f" - {name} -")
        if cfg_dict is None:
            logger.info("not provided")
        else:
            for key, val in cfg_dict.items():
                logger.info(f"{key}: {val}")
        logger.info("")


def save_sample(sample: OFData, energies: Energies, samplefile: Path) -> None:
    """Persist the optimized sample to disk after removing intermediate data."""

    sample.delete_item("overlap_matrix")
    sample.delete_item("coulomb_matrix")
    sample.delete_item("nuclear_attraction_vector")
    if hasattr(sample, "ao"):
        sample.delete_item("ao")
    # Transform back to untransformed basis
    sample = ApplyBasisTransformation()(
        sample,
        sample.transformation_matrix,
        sample.inv_transformation_matrix,
        invert=True,
    )
    sample.delete_item("transformation_matrix")
    sample.delete_item("inv_transformation_matrix")
    sample.delete_item("pred_diff")
    sample = sample.to("cpu")
    sample.add_item("energies", energies, Representation.NONE)
    torch.save(
        sample,
        samplefile,
    )


def add_sad_kwargs(ofdft_kwargs: dict, basis_info: BasisInfo) -> None:
    """Inject SAD initialization options into the OFDFT configuration."""

    sad_guess_kwargs = dict(
        dataset_statistics=DatasetStatistics(DEFAULT_DATASET_STATISTICS_PATH),
        normalization_mode=SADNormalizationMode.PER_ATOM_WEIGHTED,
        basis_info=basis_info,
        weigher_key="ground_state_only",
        spherical_average=True,
    )
    ofdft_kwargs["sad_guess_kwargs"] = sad_guess_kwargs


def run_ofdft(
    base_config: BaseConfig,
    model_config: ModelConfig,
    optimizer_config: OptimizerConfig,
    logger_format: str | None = None,
) -> list[OFDFTRunResult]:
    """Run OFDFT calculations using dataclass-based CLI configuration.

    This helper instantiates the sample generator, optimizer, and set of input
    structures from the provided configuration dataclasses before delegating to
    :func:`run_ofdft_from_components`. It is the high-level entry point exposed
    to the command line interface but remains usable programmatically for
    automated workflows.

    Args:
        base_config: Settings describing the molecular inputs and run options.
        model_config: Parameters identifying and configuring the trained model.
        optimizer_config: Hyperparameters controlling the density optimizer.
        logger_format: Optional loguru format string for auxiliary log files.

    Returns:
        A list of :class:`OFDFTRunResult` objects, one for each processed XYZ
        structure.
    """

    sample_generator = get_sample_generator_from_model_args(model_config)
    optimizer = get_optimizer_from_optimizer_args(optimizer_config)
    xyzfiles = get_xyzfiles_from_base_args(base_config)

    return run_ofdft_from_components(
        base_config=base_config,
        sample_generator=sample_generator,
        optimizer=optimizer,
        xyzfiles=xyzfiles,
        model_config=model_config,
        optimizer_config=optimizer_config,
        logger_format=logger_format,
    )


def run_ofdft_from_components(
    *,
    base_config: BaseConfig,
    sample_generator: SampleGenerator,
    optimizer: Optimizer,
    xyzfiles: Sequence[Path],
    model_config: ModelConfig | Mapping[str, Any] | None = None,
    optimizer_config: OptimizerConfig | Mapping[str, Any] | None = None,
    logger_format: str | None = None,
) -> list[OFDFTRunResult]:
    """Run OFDFT calculations using pre-instantiated dependencies.

    This function executes the end-to-end OFDFT workflow given a set of already
    constructed components. It is suitable for library use where callers manage
    model loading, optimizer creation, or input discovery themselves while
    reusing the orchestration logic implemented for the CLI.

    Args:
        base_config: Run configuration that supplies shared OFDFT options.
        sample_generator: Prepared :class:`SampleGenerator` instance.
        optimizer: Optimizer responsible for electron density updates.
        xyzfiles: Sequence of XYZ input paths to be processed.
        model_config: Optional model configuration for logging purposes.
        optimizer_config: Optional optimizer configuration for logging.
        logger_format: Optional loguru format string for per-molecule logs.

    Returns:
        A list of :class:`OFDFTRunResult` objects containing energies, density
        coefficients, and bookkeeping data for each input structure.
    """

    log_config(base_config, model_config, optimizer_config, logger)

    ofdft_kwargs = {
        key: getattr(base_config, key)
        for key in OFDFT_KWARGS
        if hasattr(base_config, key) and getattr(base_config, key) is not None
    }
    ofdft_kwargs.setdefault("disable_printing", True)
    ofdft_kwargs.setdefault("disable_pbar", False)

    if base_config.initialization == "sad":
        add_sad_kwargs(ofdft_kwargs, sample_generator.basis_info)

    results: list[OFDFTRunResult] = []
    log_format = logger_format or "{message}"

    for xyz_file in xyzfiles:
        logfile = xyz_file.with_suffix(".log")
        samplefile = xyz_file.with_suffix(".pt")
        file_logger_id = logger.add(
            logfile,
            format=log_format,
            colorize=True,
            enqueue=True,
            mode="w",
        )
        logger.info(f"Running OFDFT calculation for {xyz_file.name}\n")
        mol = gto.M(atom=str(xyz_file), charge=base_config.charge, unit="Angstrom")

        logger.debug(f"Molecule\nCharge: {mol.charge}\n{mol.tostring(format='xyz')}\n")

        log_config(
            base_config,
            model_config,
            optimizer_config,
            logger.bind(file_only=True),
        )

        energies, coeffs, converged, sample = cast(
            tuple[Energies, torch.Tensor, bool, OFData],
            run_singlepoint_ofdft(
                mol,
                sample_generator,
                sample_generator.get_functional_factory(),
                optimizer,
                initial_guess_str=base_config.initialization,
                ofdft_kwargs=ofdft_kwargs,
                return_sample=True,
            ),
        )

        if converged:
            logger.success(f"OFDFT calculation converged.\n{energies}")
        else:
            logger.critical("OFDFT calculation did not converge")

        sample_path = samplefile if base_config.save_result else None
        if sample_path is not None:
            save_sample(sample, energies, sample_path)

        results.append(
            OFDFTRunResult(
                xyz_path=xyz_file,
                energies=energies,
                converged=converged,
                coeffs=coeffs,
                sample=sample,
                sample_path=sample_path,
            )
        )

        logger.remove(file_logger_id)

    return results


def main() -> None:
    """Main function to run OFDFT calculations from the command line."""
    logger.remove()
    logger_format = "{message}"
    syslogger_id = logger.add(
        lambda msg: tqdm.write(msg, end=""),
        format=logger_format,
        colorize=True,
        enqueue=True,
        filter=console_filter,
    )

    parser = argparse.ArgumentParser(
        description="Run OFDFT calculations using MLDFT.",
    )
    parser.add_argument(
        type=Path,
        nargs="+",
        metavar="XYZFILE",
        dest="base_xyzfile",
        help="Path to an XYZ file or directory containing XYZ files.",
    )
    parser.add_argument(
        "-c",
        "--charge",
        type=int,
        default=0,
        metavar="CHARGE",
        dest="base_charge",
        help="charge of the molecule, only even numbers of electrons are supported ",
    )
    parser.add_argument(
        "--initialization",
        type=str,
        default="minao",
        choices=["minao", "sad", "hueckel"],
        dest="base_initialization",
        help="Method for initializing the electron density.",
    )
    parser.add_argument(
        "--no-normalized-initialization",
        action="store_false",
        dest="base_normalize_initial_guess",
        help="Whether to normalize the initial guess to the correct number of electrons.",
    )
    parser.add_argument(
        "--no-save",
        action="store_false",
        dest="base_save_result",
        help="Whether to save the resulting sample to a .pt file.",
    )

    model_parser = parser.add_argument_group("Model and functional configs")
    model_parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        metavar="MODELNAME or PATH",
        dest="model_model",
        help="Path to a trained model or either 'str25_qm9' or 'str25_qmugs' if the setup script has been called",
    )

    model_parser.add_argument(
        "--not-use-last-ckpt",
        action="store_false",
        dest="model_use_last_ckpt",
        help="Whether to use the last checkpoint in the model directory.",
    )
    model_parser.add_argument(
        "--device",
        type=str,
        metavar="DEVICE",
        dest="model_device",
        default="cpu" if not torch.cuda.is_available() else "cuda",
        help="Device to run the calculations on (e.g., 'cpu' or 'cuda').",
    )
    model_parser.add_argument(
        "--transform-device",
        type=str,
        default="cpu",
        metavar="TRANSFORM_DEVICE",
        dest="model_transform_device",
        help="Device to perform the transformations on (e.g., 'cpu' or 'cuda').",
    )
    model_parser.add_argument(
        "--negative-integrated-density-penalty-weight",
        type=float,
        default=0.0,
        metavar="WEIGHT",
        dest="model_negative_integrated_density_penalty_weight",
        help="Weight for the penalty term for negative integrated density.",
    )

    optimizer_parser = parser.add_argument_group("Density optimizer configs")

    optimizer_parser.add_argument(
        "--optimizer",
        type=str,
        default="gradient-descent-torch",
        choices=["gradient-descent", "vector-adam", "gradient-descent-torch"],
        metavar="OPTIMIZER",
        dest="optimizer_optimizer",
        help="Optimizer to use for density optimization.",
    )
    optimizer_parser.add_argument(
        "--max-cycle",
        type=int,
        default=10000,
        metavar="MAXCYCLE",
        dest="optimizer_max_cycle",
        help="Maximum number of iterations for the optimizer.",
    )
    optimizer_parser.add_argument(
        "--convergence-tolerance",
        type=float,
        default=1e-4,
        dest="optimizer_convergence_tolerance",
        metavar="TOL",
        help="Convergence tolerance for the optimizer.",
    )
    optimizer_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        metavar="LR",
        dest="optimizer_lr",
        help="Learning rate for the optimizer.",
    )
    optimizer_parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="MOMENTUM",
        dest="optimizer_momentum",
        help="Momentum for the optimizer (gradient-descent-torch).",
    )
    optimizer_parser.add_argument(
        "--betas",
        nargs=2,
        type=float,
        default=(0.9, 0.999),
        dest="optimizer_betas",
        metavar=("BETA1", "BETA2"),
        help="Betas for the Adam optimizer.",
    )

    args = parser.parse_args(sys.argv[1:])

    base_args = extract_group("base", args, BaseConfig)
    model_args = extract_group("model", args, ModelConfig)
    optimizer_args = extract_group("optimizer", args, OptimizerConfig)

    logger.info(CLI_HEADER)

    logger.info("Parsed command line arguments:")
    logger.info(f"MLDFT called with:\n{' '.join(sys.argv)}\n")
    run_ofdft(base_args, model_args, optimizer_args, logger_format)


if __name__ == "__main__":
    main()
