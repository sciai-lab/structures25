import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, cast

import torch  # type: ignore[import]

from mldft.ofdft.optimizer import GradientDescent, TorchOptimizer, VectorAdam
from mldft.ofdft.run_density_optimization import SampleGenerator

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class BaseConfig:
    """Configuration required to construct the base OFDFT inputs."""

    xyzfile: tuple[str | Path, ...]
    charge: int = 0
    initialization: str = "minao"
    normalize_initial_guess: bool = True
    save_result: bool = True
    ks_basis: str | None = None
    proj_minao_module: str | None = None
    sad_guess_kwargs: dict | None = None
    disable_printing: bool | None = None

    def __post_init__(self) -> None:
        self.xyzfile = tuple(Path(path) for path in self.xyzfile)


@dataclass
class ModelConfig:
    """Configuration describing the trained MLDFT model to use."""

    model: str
    use_last_ckpt: bool = True
    device: str = DEFAULT_DEVICE
    transform_device: str = "cpu"
    negative_integrated_density_penalty_weight: float = 0.0


@dataclass
class OptimizerConfig:
    """Configuration controlling the density optimization routine."""

    optimizer: str = "gradient-descent-torch"
    max_cycle: int = 10000
    convergence_tolerance: float = 1e-4
    lr: float = 1e-3
    momentum: float = 0.9
    betas: Sequence[float] = (0.9, 0.999)

    def __post_init__(self) -> None:
        self.betas = tuple(self.betas)


def get_runpath(name: str) -> Path:
    """Get the path to a named model."""
    dft_models = os.environ.get("DFT_MODELS")
    if dft_models is None:
        raise ValueError(
            "Environment variable DFT_MODELS not set. Please set it to the directory containing the models."
        )
    dft_models = Path(dft_models)
    run_path = dft_models / "train/runs" / f"{name}"
    return run_path


NAMED_MODELS = {
    "str25_qm9": get_runpath("trained-on-qm9"),
    "str25_qmugs": get_runpath("trained-on-qmugs"),
}


def get_gradient_descent_optimizer(optimizer_args: OptimizerConfig) -> GradientDescent:
    """Instantiate a simple gradient descent optimizer."""
    return GradientDescent(
        learning_rate=optimizer_args.lr,
        max_cycle=optimizer_args.max_cycle,
        convergence_tolerance=optimizer_args.convergence_tolerance,
    )


def get_gradient_descent_torch_optimizer(
    optimizer_args: OptimizerConfig,
) -> TorchOptimizer:
    """Instantiate a gradient descent optimizer using PyTorch's SGD."""
    return TorchOptimizer(
        torch.optim.SGD,
        lr=optimizer_args.lr,
        momentum=optimizer_args.momentum,
        max_cycle=optimizer_args.max_cycle,
        convergence_tolerance=optimizer_args.convergence_tolerance,
    )


def get_vector_adam_optimizer(optimizer_args: OptimizerConfig) -> VectorAdam:
    """Instantiate a Vector Adam optimizer."""
    return VectorAdam(
        learning_rate=optimizer_args.lr,
        betas=cast("tuple[float, float]", tuple(optimizer_args.betas)),
        max_cycle=optimizer_args.max_cycle,
        convergence_tolerance=optimizer_args.convergence_tolerance,
    )


OPTIMIZER_CHOICES = {
    "gradient-descent": get_gradient_descent_optimizer,
    "gradient-descent-torch": get_gradient_descent_torch_optimizer,
    "vector-adam": get_vector_adam_optimizer,
}


def get_optimizer_from_optimizer_args(
    optimizer_args: OptimizerConfig,
) -> GradientDescent | TorchOptimizer | VectorAdam:
    """Instantiate an optimizer from optimizer arguments."""
    optimizer_name = optimizer_args.optimizer
    if optimizer_name not in OPTIMIZER_CHOICES:
        raise ValueError(
            f"Unknown optimizer {optimizer_name}. Choose from {list(OPTIMIZER_CHOICES.keys())}."
        )

    return OPTIMIZER_CHOICES[optimizer_name](optimizer_args)


def get_sample_generator_from_model_args(
    model_args: ModelConfig,
) -> SampleGenerator:
    """Instantiate a SampleGenerator from model arguments."""

    model_name = model_args.model
    if model_name in NAMED_MODELS:
        run_path = NAMED_MODELS[model_name]
    else:
        run_path = Path(model_name)

    sample_generator = SampleGenerator.from_run_path(
        run_path,
        device=model_args.device,
        transform_device=model_args.transform_device,
        negative_integrated_density_penalty_weight=(
            model_args.negative_integrated_density_penalty_weight
        ),
        use_last_ckpt=model_args.use_last_ckpt,
    )

    return sample_generator


def get_xyzfiles_from_base_args(base_args: BaseConfig) -> list[Path]:
    """Get a list of XYZ files from base arguments."""
    xyzfiles = []
    for path in base_args.xyzfile:
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist.")
        if path.is_file() and path.suffix == ".xyz":
            xyzfiles.append(path)
        elif path.is_dir():
            xyzfiles.extend(sorted(path.glob("*.xyz")))

    return xyzfiles
