from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig

from mldft.utils.plotting.summary_density_optimization import (
    density_optimization_summary_pdf_plot,
    get_runwise_density_optimization_data,
    save_density_optimization_metrics,
)


@hydra.main(version_base="1.3", config_path="../../configs/ofdft", config_name="plot_ofdft.yaml")
def plot_from_dir(cfg: DictConfig):
    """Plot the density optimization summary based on a directory of samples."""
    if cfg.get("plot_l1_norm"):
        logger.warning("L1 norm will be computed, for QMUGS this should be off by default.")
    run_data_dict = get_runwise_density_optimization_data(
        sample_dir=Path(cfg.get("sample_dir")),
        n_molecules=cfg.get("n_molecules"),
        plot_l1_norm=cfg.get("plot_l1_norm"),
    )

    if cfg.out_pdf_path is None:
        out_pdf_path = Path(cfg.get("sample_dir")).parent / "density_optimization_summary.pdf"
    else:
        out_pdf_path = Path(cfg.get("out_pdf_path"))

    save_density_optimization_metrics(
        output_path=out_pdf_path.parent / "density_optimization_metrics.yaml",
        run_data_dict=run_data_dict,
    )

    density_optimization_summary_pdf_plot(
        out_pdf_path=out_pdf_path,
        run_data_dict=run_data_dict,
        subsample=cfg.get("swarm_plot_subsample"),
    )


if __name__ == "__main__":
    plot_from_dir()
