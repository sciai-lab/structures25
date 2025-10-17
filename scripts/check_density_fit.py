import argparse

import numpy as np
import zarr
from loguru import logger
from pyscf import dft
from tqdm import tqdm

from mldft.datagen.methods.density_fitting import ksdft_density_matrix
from mldft.ofdft.basis_integrals import get_normalization_vector
from mldft.utils.environ import get_mldft_data_path
from mldft.utils.grids import compute_l1_norm_orbital_vs_density_basis
from mldft.utils.molecules import (
    build_mol_with_even_tempered_basis,
    build_molecule_np,
    load_scf,
)
from mldft.utils.pyscf_pretty_print import mole_to_sum_formula

filename_map = {"QMUGS": "qmugs", "QM9": "qm9"}


def check_density_fit(dataset: str):
    """Short script to check the densities of a dataset."""
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
    data_dir = get_mldft_data_path()
    label_dir = data_dir / dataset / "labels"
    kohn_sham_dir = data_dir / dataset / "kohn_sham"
    label_files = list(sorted(label_dir.glob("*.zarr.zip")))
    filename = filename_map[dataset]
    for label_file in tqdm(label_files):
        bad_density = False
        logger.info(f"Checking file {label_file}.")
        label_data = zarr.open(str(label_file))
        coeffs = label_data["of_labels/spatial/coeffs"][-1]
        kohn_sham_file = kohn_sham_dir / f"{filename}_{label_file.stem.split('.')[0]}.chk"
        results, initialization, data_of_iteration = load_scf(kohn_sham_file)
        kohn_sham_data = data_of_iteration[-1]
        mo_coeff = kohn_sham_data["molecular_coeffs_orbitals"]
        mo_occ = kohn_sham_data["occupation_numbers_orbitals"]
        gamma = ksdft_density_matrix(mo_coeff, mo_occ)
        atomic_numbers = label_data["geometry/atomic_numbers"]
        n_electron = sum(atomic_numbers)
        mol_ks = build_molecule_np(atomic_numbers, label_data["geometry/atom_pos"])
        mol_density = build_mol_with_even_tempered_basis(mol_ks)
        logger.info(
            f"Computing atom {mole_to_sum_formula(mol_ks)} with {len(atomic_numbers)} atoms"
        )
        grid = dft.Grids(mol_ks)
        grid.level = 2
        grid.build()
        l1_norm = compute_l1_norm_orbital_vs_density_basis(
            mol_ks, mol_density, grid, gamma, coeffs
        )
        l1_norm_per_electron = l1_norm / n_electron
        basis_integrals = get_normalization_vector(mol_density)
        n_electrons_density = coeffs @ basis_integrals
        if l1_norm_per_electron > 2e-3:
            logger.warning(f"Large L1 density error per electron: \t {l1_norm_per_electron}")
            bad_density = True
        if np.abs(n_electron - n_electrons_density) > 1e-2:
            logger.warning(f"Electron count mismatch: {n_electron} vs {n_electrons_density}")
            bad_density = True
        logger.info(f"L1 density error: {l1_norm}")
        logger.info(f"L1 density error per electron: {l1_norm_per_electron}")
        logger.info(f"Electrons in molecule: {n_electron}")
        logger.info(f"Electrons in fitted density: {n_electrons_density}")
        if bad_density:
            # Write bad files to disk
            logger.warning("Writing bad density to disk.")
            output_file = data_dir / dataset / "bad_densities.yaml"
            with open(output_file, "a") as f:
                f.write(f"{label_file}:\n")
                f.write(f"  l1_norm: {l1_norm}\n")
                f.write(f"  l1_norm_per_electron: {l1_norm_per_electron}\n")
                f.write(f"  n_electron: {n_electron}\n")
                f.write(f"  n_electrons_density: {n_electrons_density}\n")
                f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="QMUGS")
    args = parser.parse_args()
    check_density_fit(args.dataset)
